#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com


# fintune on alpaca
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python post_training.py \
      --prune_model prune_log/llama_prune/your_pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir tune_log_llama2/alpaca \
      --wandb_project alpaca 

