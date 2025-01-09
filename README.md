# Full_sft_llama
This is a repo that use FSDP to do fully supervised fine-tuning on Llama3-8B

## Running
torchrun --nproc_per_node=4 fully_ft_llama3.py \
         --train_data_path DATASET_PATH \
         --output_path OUTPUT_PATH
