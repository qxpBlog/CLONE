#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_LOG_PATH=/home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/eval_log


mkdir -p "$SAVE_LOG_PATH"

# channel 
echo "channel eval gsm"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
CUDA_VISIBLE_DEVICES=2 python gsm8k.py \
    --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
    --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_channel \
    --finetuning_type full \
    --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/channel \
    > "$SAVE_LOG_PATH/0.2_channel.log" 2>&1

# 20%
echo "20% eval gsm"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
CUDA_VISIBLE_DEVICES=2 python gsm8k.py \
    --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2/pytorch_model.bin \
    --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2 \
    --finetuning_type full \
    --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/0.2_re \
    > "$SAVE_LOG_PATH/0.2.log" 2>&1

