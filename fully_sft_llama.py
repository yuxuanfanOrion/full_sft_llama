import torch
import os
import json
import wandb
import functools
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LLaMA model")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the model outputs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8B",
        help="Name or path of the base model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    return parser.parse_args()

def setup_distributed_env():
    """Set up distributed training environment variables"""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

def setup_fsdp():
    """Configure FSDP parameters"""
    setup_distributed_env()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    torch.distributed.init_process_group(backend="nccl")
    
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        ),
        "backward_prefetch": BackwardPrefetch.BACKWARD_POST,
        "cpu_offload": CPUOffload(offload_params=True),
    }
    
    return fsdp_config

class TrainingConfig:
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset_name = args.train_data_path
        
        # Training configuration
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = 2e-5
        self.max_length = 512
        self.gradient_accumulation_steps = 8
        self.weight_decay = 0.01
        self.warmup_steps = 100
        
        # Output configuration
        self.output_dir = args.output_path
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500

class SafeDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
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
    
    def disable_inplace(model):
        def disable_module_inplace(module):
            if hasattr(module, 'inplace'):
                module.inplace = False
        model.apply(disable_module_inplace)
    
    disable_inplace(model)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = model.to(f"cuda:{local_rank}")
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    with open(config.dataset_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    def format_data(data_split):
        formatted_data = []
        for sample in data_split:
            prompt = sample['prompt']
            ground_truth = sample['ground_truth']
            answer = ground_truth[0] if ground_truth else ""
            full_text = f"Below is an instruction. Write a response that completes the instruction.\n\nInstruction: Solve this math problem: {prompt}\nResponse: {answer}"
            formatted_data.append({"text": full_text})
        return formatted_data

    train_dataset = Dataset.from_list(format_data(train_data))
    val_dataset = Dataset.from_list(format_data(val_data))
    
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
        
        return tokenized

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False
    )
    
    return {
        "train": tokenized_train,
        "validation": tokenized_val
    }

def train(config, model, tokenized_datasets, tokenizer):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "activation_checkpointing": True,
            "forward_prefetch": True,
            "backward_prefetch": "BACKWARD_POST",
            "limit_all_gathers": True,
            "sync_module_states": True,
            "use_orig_params": True
        },
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    data_collator = SafeDataCollator(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()

def main():
    args = parse_args()
    config = TrainingConfig(args)
    
    # Initialize wandb only on main process
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        wandb.init(project="llama3-full-finetune")
    
    fsdp_config = setup_fsdp()
    model, tokenizer = prepare_model_and_tokenizer(config)
    tokenized_datasets = prepare_dataset(config, tokenizer)
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **fsdp_config
    )
    
    train(config, model, tokenized_datasets, tokenizer)

if __name__ == "__main__":
    main()
