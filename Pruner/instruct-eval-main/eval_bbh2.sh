#!/bin/bash 
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_LOG_DIR=/home/iotsc01/xinpengq/LLM-Pruner-main/eval_logs_bbh



mkdir -p "$SAVE_LOG_DIR"
# base model

# echo "our tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/our \
#     > "$SAVE_LOG_DIR/our_tuned.log" 2>&1

# # 20%

# echo "20%"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/0.2.log" 2>&1

# echo "20% tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2 \
#     > "$SAVE_LOG_DIR/0.2_tuned.log" 2>&1

# Random

echo "random"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_random/pytorch_model.bin \
    > "$SAVE_LOG_DIR/random.log" 2>&1

echo "random tuned"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_random/pytorch_model.bin \
    --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_random \
    > "$SAVE_LOG_DIR/random_tuned.log" 2>&1


echo "our"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
    > "$SAVE_LOG_DIR/our.log" 2>&1
