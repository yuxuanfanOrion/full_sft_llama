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
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
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
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
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
        default="meta-llama/Llama-3.1-8B",
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
    
    parser.add_argument(
        "--freeze_mlp_layers",
        type=str,
        default="",
        help="Comma-separated list of MLP layer indices to freeze (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--freeze_attention_layers",
        type=str,
        default="",
        help="Comma-separated list of attention layer indices to freeze (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--freeze_attention_heads",
        type=str,
        default="",
        help="Comma-separated list of 'layer_idx:head_idx' to freeze specific attention heads (e.g., '0:0,0:1,1:2' freezes heads 0,1 in layer 0 and head 2 in layer 1)"
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
        "use_orig_params": True,
    }
    
    return fsdp_config

class CustomAttentionHeadMask(torch.nn.Module):
    """Custom module to mask specific attention heads during forward pass"""
    def __init__(self, original_attn, frozen_head_indices):
        super().__init__()
        self.original_attn = original_attn
        self.frozen_head_indices = frozen_head_indices
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        
        # Create a mask for the frozen heads
        self.head_mask = torch.ones(self.num_heads, dtype=torch.bool)
        for head_idx in frozen_head_indices:
            self.head_mask[head_idx] = False
            
    def forward(self, *args, **kwargs):
        output = self.original_attn(*args, **kwargs)
        
        if isinstance(output, tuple):
            attn_output = output[0]
            batch_size, seq_length, _ = attn_output.size()
            attn_output = attn_output.view(batch_size, seq_length, self.num_heads, -1)
            
            head_mask = self.head_mask.to(attn_output.device)
            attn_output = attn_output * head_mask.view(1, 1, -1, 1)
            
            attn_output = attn_output.view(batch_size, seq_length, -1)
            return (attn_output,) + output[1:]
        return output

def selectively_freeze_layers(model, freeze_mlp_layers, freeze_attention_layers, freeze_attention_heads):
    """Freeze specified MLP and attention layers/heads"""
    if not isinstance(model, (LlamaForCausalLM, FSDP)):
        return

    # Convert layer indices from string to list of integers
    mlp_layers = [int(x.strip()) for x in freeze_mlp_layers.split(',')] if freeze_mlp_layers else []
    attention_layers = [int(x.strip()) for x in freeze_attention_layers.split(',')] if freeze_attention_layers else []
    head_map = parse_head_spec(freeze_attention_heads)

    # Access the decoder layers
    if isinstance(model, FSDP):
        decoder_layers = model.module.model.layers
    else:
        decoder_layers = model.model.layers

    for idx, layer in enumerate(decoder_layers):
        # Freeze MLP
        if idx in mlp_layers:
            freeze_layer_params(layer.mlp)
            print(f"Freezing MLP layer {idx}")

        # Freeze entire attention layer
        if idx in attention_layers:
            freeze_layer_params(layer.self_attn)
            print(f"Freezing entire Attention layer {idx}")
        
        # Freeze specific attention heads
        if idx in head_map:
            frozen_heads = head_map[idx]
            original_attn = layer.self_attn
            
            # 计算head相关的维度
            num_heads = original_attn.num_heads
            head_dim = original_attn.head_dim
            
            # Replace with custom attention module
            layer.self_attn = CustomAttentionHeadMask(original_attn, frozen_heads)
            
            # Freeze the parameters for specific heads
            for head_idx in frozen_heads:
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # Freeze the corresponding parts of the Q, K, V projections
                if hasattr(original_attn, 'q_proj'):
                    original_attn.q_proj.weight.data[start_idx:end_idx].requires_grad_(False)
                    original_attn.k_proj.weight.data[start_idx:end_idx].requires_grad_(False)
                    original_attn.v_proj.weight.data[start_idx:end_idx].requires_grad_(False)
                    original_attn.o_proj.weight.data[:, start_idx:end_idx].requires_grad_(False)
                
                print(f"Freezing Attention head {head_idx} in layer {idx}")

def parse_head_spec(head_spec):
    """Parse head specification string into a dict of {layer_idx: [head_indices]}"""
    if not head_spec:
        return {}
    
    head_map = {}
    specs = head_spec.split(',')
    for spec in specs:
        layer_idx, head_idx = map(int, spec.split(':'))
        if layer_idx not in head_map:
            head_map[layer_idx] = []
        head_map[layer_idx].append(head_idx)
    return head_map

def freeze_layer_params(module):
    """Freeze parameters of a module"""
    for param in module.parameters():
        param.requires_grad = False


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

def prepare_model_and_tokenizer(config, args):
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
    
    # Freeze specified layers before moving to GPU
    selectively_freeze_layers(
        model, 
        args.freeze_mlp_layers, 
        args.freeze_attention_layers,
        args.freeze_attention_heads
    )
    
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
    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        wandb.init(project="llama3-full-finetune")
    
    fsdp_config = setup_fsdp()
    model, tokenizer = prepare_model_and_tokenizer(config, args)
    
    # 先包装FSDP
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **fsdp_config
    )
    
    # 然后再进行冻结操作
    selectively_freeze_layers(
        model, 
        args.freeze_mlp_layers, 
        args.freeze_attention_layers,
        args.freeze_attention_heads
    )
    
    tokenized_datasets = prepare_dataset(config, tokenizer)
    train(config, model, tokenized_datasets, tokenizer)

if __name__ == "__main__":
    main()
