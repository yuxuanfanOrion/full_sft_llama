import torch
import os
import json
import wandb
from datetime import datetime
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset, load_dataset, DatasetDict
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LLaMA model")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data JSON file"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the test data JSON file"
    )
    parser.add_argument(
        "--output_path",
        default="./output/",
        type=str,
        help="Path to save the model outputs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Name or path of the base model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    
    
    
    
    return parser.parse_args()

args = parse_args()


class TrainingConfig:
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset_name = args.train_data_path
        self.test_data_path = args.test_data_path
        
        # Training configuration
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.max_length = 20
        self.gradient_accumulation_steps = 1
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.max_test_samples = None
        
        # Output configuration
        self.output_dir = args.output_path
        self.logging_steps = 150
        self.save_steps = 150
        self.eval_steps = 150

class SafeDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        
        # 确保输入包含所需的键
        if "input_ids" not in features[0]:
            raise KeyError("Missing 'input_ids' in features")
        
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [ids.copy() for ids in input_ids]
        
        max_length = max(len(ids) for ids in input_ids)
        
        def pad_sequence(sequences, pad_value):
            padded_sequences = []
            for seq in sequences:
                padding_length = max_length - len(seq)
                padded_seq = seq + [pad_value] * padding_length
                padded_sequences.append(padded_seq)
            return torch.tensor(padded_sequences, dtype=torch.long)
        
        batch["input_ids"] = pad_sequence(input_ids, self.tokenizer.pad_token_id)
        batch["attention_mask"] = pad_sequence(attention_mask, 0)
        batch["labels"] = pad_sequence(labels, -100)
        
        return batch

def prepare_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    

    if torch.cuda.is_available():
        model = model.to("cuda")
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    # 加载单一的 JSON 文件
    dataset = load_dataset('json', data_files=config.dataset_name, split='train')
    
    # 按照 95% 训练集和 5% 验证集进行分割
    split_ratio = 0.95
    train_val = dataset.train_test_split(test_size=1 - split_ratio, seed=42)
    
    # 将分割后的数据集转换为 DatasetDict 格式
    tokenized_datasets = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test']
    })
    
    def format_data(example):
        prompt = example['prompt']
        ground_truth = example['ground_truth']
        answer = ground_truth[0] if ground_truth else ""
        return {"text": f"{prompt}{answer}"}
    
    # 对训练和验证集应用格式化函数
    tokenized_datasets = tokenized_datasets.map(
        format_data,
        remove_columns=tokenized_datasets['train'].column_names,
        num_proc=4  # 使用多进程加速
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=config.max_length,
        )
    
    # 对训练和验证集应用分词函数
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,  # 使用多进程加速
        load_from_cache_file=True  # 启用缓存
    )
    
    return tokenized_datasets


def prepare_test_dataset(config, tokenizer):
    if not config.test_data_path:
        return None
        
    with open(config.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if config.max_test_samples:
        test_data = test_data[:config.max_test_samples]
    
    def format_test_data(data_split):
        formatted_data = []
        for sample in data_split:
            prompt = sample['prompt']
            ground_truth = sample['ground_truth']
            answer = ground_truth[0] if ground_truth else ""
            full_text = f"{prompt}{answer}"
            formatted_data.append({
                "text": full_text,
                "prompt": prompt,
                "ground_truth": ground_truth
            })
        return formatted_data

    test_dataset = Dataset.from_list(format_test_data(test_data))
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=config.max_length,
            return_tensors=None
        )
        
        tokenized["input_ids"] = [list(ids) for ids in tokenized["input_ids"]]
        tokenized["attention_mask"] = [list(mask) for mask in tokenized["attention_mask"]]
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        
        # 保留原始信息用于评估
        tokenized["prompt"] = examples["prompt"]
        tokenized["ground_truth"] = examples["ground_truth"]
        
        return tokenized

    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False
    )
    
    return tokenized_test


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = kwargs.get('compute_metrics')
        # 确保有label_names
        if not hasattr(self, 'label_names'):
            self.label_names = ['labels']

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits
                
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean().detach()
                
                if prediction_loss_only:
                    return (loss, None, None)
                
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach()
                    if len(logits.shape) == 3:
                        logits = logits.argmax(dim=-1)
                
                # 直接从inputs中获取labels
                labels = inputs.get('labels')
                if labels is not None:
                    labels = labels.detach()
                
                return (loss, logits, labels)
                
        except Exception as e:
            print(f"Error in prediction_step: {str(e)}")
            print(f"Input keys: {inputs.keys()}")
            print(f"Label names: {self.label_names}")
            raise

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        
        if num_items_in_batch is not None:
            loss = loss * inputs["labels"].shape[0] / num_items_in_batch
            
        return (loss, outputs) if return_outputs else loss

def create_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        print("\n=== Starting compute_metrics ===")
        try:
            predictions, labels = eval_preds
            if predictions is None or labels is None:
                print("Warning: No predictions or labels available")
                return {}

            # 解码预测和标签
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            def extract_number_after_equals(text):
                # 找到等号后的数字
                if '=' in text:
                    after_equals = text.split('=')[1].strip()
                    try:
                        # 提取第一个数字
                        number = after_equals.split()[0]
                        return float(number)
                    except:
                        return None
                return None
            
            results = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "details": [],
                "metrics": {}
            }

            # 计算准确率
            correct_count = 0
            total_count = 0
            
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_num = extract_number_after_equals(pred)
                #print(f"pred_num: {pred_num}")
                label_num = extract_number_after_equals(label)
                #print(f"label_num: {label_num}")
                
                is_correct = False
                if pred_num is not None and label_num is not None:
                    is_correct = abs(pred_num - label_num) < 1e-6
                    if is_correct:
                        correct_count += 1
                total_count += 1
                
                # 记录每个样本的详细信息
                results["details"].append({
                    "decoded_pred": pred,
                    "decoded_label": label,
                    "pred_num": pred_num,
                    "label_num": label_num,
                    "is_correct": is_correct
                })
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            results["metrics"] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count
            }
            
            # 保存结果到json文件
            os.makedirs(args.output_path, exist_ok=True)
            output_file = os.path.join(
                args.output_path, 
                f"metrics_details_{results['timestamp']}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDetailed results saved to: {output_file}")
            print(f"Accuracy: {accuracy:.4f}")
            
            return {"accuracy": accuracy}
            
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
            
    return compute_metrics

    
def test_model(model, test_dataset, tokenizer, config, device):
    if test_dataset is None:
        print("No test dataset provided")
        return
    
    model.eval()
    test_results = []
    correct_count = 0
    total_count = 0
    
    def extract_number_after_equals(text):
        # 找到等号后的数字
        if '=' in text:
            after_equals = text.split('=')[1].strip()
            try:
                # 提取第一个数字
                number = after_equals.split()[0]
                return float(number)
            except:
                return None
        return None
    
    for i in range(0, len(test_dataset), config.batch_size):
        batch = test_dataset[i:i + config.batch_size]
        
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]
        
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=30,
                num_beams=4,
                temperature=0.7,  # 添加温度参数
                top_p=0.9,       # 添加top_p采样    
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for prompt, ground_truth, generated in zip(prompts, ground_truths, generated_texts):
            generated_answer = extract_number_after_equals(generated)
            true_answer = ground_truth
            
            is_correct = False
            if generated_answer is not None and true_answer is not None:
                is_correct = abs(generated_answer - float(true_answer[0])) < 1e-6
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth[0],
                "generated": generated,
                "generated_answer": generated_answer,
                "true_answer": true_answer,
                "is_correct": is_correct
            }
            test_results.append(result)
    
    # 计算并打印最终准确率
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    # 保存测试结果
    test_results_with_metrics = {
        "results": test_results,
        "metrics": {
            "accuracy": final_accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count
        }
    }
    
    output_file = os.path.join(config.output_dir, "test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_results_with_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to {output_file}")

def train(config, model, tokenized_datasets, tokenizer, compute_metrics_fn):
    data_collator = SafeDataCollator(tokenizer)
    print("\nValidating datasets:")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    print(f"Validation dataset size: {len(tokenized_datasets.get('validation', []))}")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="no",
        prediction_loss_only=False,
        dataloader_pin_memory=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
        bf16=True,
        remove_unused_columns=False,
    )
    
    class DebugCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            print("\n=== Evaluation Callback ===")
            print(f"Current step: {state.global_step}")
            print(f"Received metrics: {metrics}")
            print("=========================\n")
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        callbacks=[DebugCallback()]
    )
    
    print("\nPerforming initial evaluation...")
    initial_metrics = trainer.evaluate()
    print(f"Initial evaluation metrics: {initial_metrics}")
    
    print("\nStarting training...")
    trainer.train()
    
     # 在训练结束后进行最终评估
    final_eval_results = trainer.evaluate()
    print(f"\nFinal evaluation results: {final_eval_results}")
    
    # 保存最终评估结果
    eval_output_file = os.path.join(config.output_dir, "eval_results.json")
    with open(eval_output_file, "w") as f:
        json.dump(final_eval_results, f, indent=2)
    
    #trainer.save_model()

def main():
    args = parse_args()
    config = TrainingConfig(args)
    
    # Initialize wandb only on main process
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        wandb.init(project="llama3-full-finetune")
    
    #fsdp_config = setup_fsdp()
    model, tokenizer = prepare_model_and_tokenizer(config)
    tokenized_datasets = prepare_dataset(config, tokenizer)
    compute_metrics_fn = create_compute_metrics(tokenizer)
    
    train(config, model, tokenized_datasets, tokenizer, compute_metrics_fn)
    
    # 准备测试数据
    test_dataset = prepare_test_dataset(config, tokenizer)
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载最佳模型进行测试
    #best_model_path = os.path.join(config.output_dir, "best")
    #if os.path.exists(best_model_path):
        # model = AutoModelForCausalLM.from_pretrained(
        #     best_model_path,
        #     torch_dtype=torch.bfloat16
        # ).to(device)
    
    # 进行测试
    test_model(model, test_dataset, tokenizer, config, device)

if __name__ == "__main__":
    main()
