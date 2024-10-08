#!/bin/bash 
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_LOG_DIR=/home/iotsc01/xinpengq/LLM-Pruner-main/eval_logs_bbh



mkdir -p "$SAVE_LOG_DIR"


# Random-Layer
# echo "random 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.99 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer \
#       --base_model $BASE_MODEL 

# echo "Random-Layer"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_random.log" 2>&1

# PPL-Layer
# echo "imp 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 0.99 0.99 0.0 0.0 0.99 0.0 0.0 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer \
#       --base_model $BASE_MODEL 

# echo "PPL-Layer"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_ppl.log" 2>&1


# shortGPT
# echo "shortGPT 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.0 0.0 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer \
#       --base_model $BASE_MODEL 

# echo "shortGPT"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_shortgpt.log" 2>&1


# echo "shortGPT tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/shortgpt \
#     > "$SAVE_LOG_DIR/layer_shortgpt_tuned.log" 2>&1

# random
echo "random 7 layers"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
      --pruning_ratio 0.0 0.99 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer \
      --base_model $BASE_MODEL 

export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/post_training.py \
      --prune_model /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/random_layer \
      --wandb_project llama_tune 

echo "random tuned"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
    --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/random_layer \
    > "$SAVE_LOG_DIR/layer_random_tuned.log" 2>&1

# ppl
echo "imp 7 layers"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
      --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 0.99 0.99 0.0 0.0 0.99 0.0 0.0 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer \
      --base_model $BASE_MODEL 

export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/post_training.py \
      --prune_model /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/ppl_layer \
      --wandb_project llama_tune 

echo "shortGPT tuned"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
    --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/ppl_layer \
    > "$SAVE_LOG_DIR/layer_ppl_tuned.log" 2>&1

