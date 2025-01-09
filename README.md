# Full_sft_llama
This is a repo that use FSDP to do full parameters supervised fine-tuning on Llama3-8B or freeze any mlp and attention heads to fine-tune the llama3-8b
## file
- freeze_fine-tune_llama3.py (You can freeze any MLP or attention head)
- fully_stf_llama.py 

## Running
torchrun --nproc_per_node=4 fully_Sft_llama3.py \
         --train_data_path DATASET_PATH \
         --output_path OUTPUT_PATH
