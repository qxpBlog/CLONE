#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=meta-llama/Llama-2-7b-hf
LOG_PATH=eval_logs

mkdir -p "$LOG_PATH"

echo "20% block pruning"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python hf_prune.py \
      --pruning_ratio 0.2 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name llama_prune \
      --base_model $BASE_MODEL \
      > "$LOG_PATH/llama_prune_0.2.log" 2>&1


echo "20% block fine-tuning"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python post_training.py \
      --prune_model prune_log/llama_prune/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir tune_log_llama2/0.2 \
      --wandb_project 0.2 
