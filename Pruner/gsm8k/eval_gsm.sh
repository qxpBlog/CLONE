#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_LOG_PATH=/home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/eval_log


mkdir -p "$SAVE_LOG_PATH"

# base model
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
CUDA_VISIBLE_DEVICES=3 python gsm8k_base.py \
    --model_name_or_path $BASE_MODEL \
    --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/our \
    --finetuning_type full \
    --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/base \
    > "$SAVE_LOG_PATH/base.log" 2>&1


# our 
# echo "our eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=1 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/our \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/our/ \
#     > "$SAVE_LOG_PATH/our.log" 2>&1

# # l2 
# echo "l2 eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=1 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_l2/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_l2 \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/l2 \
#     > "$SAVE_LOG_PATH/0.2_l2.log" 2>&1


# # random 
# echo "random eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=1 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_random/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_random \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/random \
#     > "$SAVE_LOG_PATH/0.2_random.log" 2>&1
    
# # channel 
# echo "channel eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=1 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_channel \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/channel \
#     > "$SAVE_LOG_PATH/0.2_channel.log" 2>&1

# # 20%
# echo "20% eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=1 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2 \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/0.2 \
#     > "$SAVE_LOG_PATH/0.2.log" 2>&1


# # our gsm
# echo "our gsm eval gsm"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main/"
# CUDA_VISIBLE_DEVICES=0 python gsm8k.py \
#     --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
#     --peft_model_path /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft/gsm \
#     --finetuning_type full \
#     --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/gsm8k/outputs/our_gsm/ \
#     > "$SAVE_LOG_PATH/our_gsm.log" 2>&1

