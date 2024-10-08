#!/bin/bash 
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_LOG_DIR=/home/iotsc01/xinpengq/LLM-Pruner-main/eval_logs_bbh



mkdir -p "$SAVE_LOG_DIR"

# (1)base model
# echo "base"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path $BASE_MODEL \
#     > "$SAVE_LOG_DIR/base.log" 2>&1

# echo "base tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path $BASE_MODEL \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/base_model \
#     > "$SAVE_LOG_DIR/base_tuned.log" 2>&1

# (2)our
# echo "our"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/our.log" 2>&1

# echo "our tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/our \
#     > "$SAVE_LOG_DIR/our_tuned.log" 2>&1

# (3)20%
# echo "20% pruning"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.2 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name llama_prune_0.2 \
#       --base_model $BASE_MODEL 

# echo "20%"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python main.py \
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

# (4)L2
# echo "l2"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_l2/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/l2.log" 2>&1

# echo "l2 tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_l2/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_l2 \
#     > "$SAVE_LOG_DIR/l2_tuned.log" 2>&1

# (5)Random
# echo "random pruning"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.2 \
#       --block_wise \
#       --pruner_type random \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name llama_prune_0.2_random \
#       --base_model $BASE_MODEL 

# echo "random"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_random/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/random.log" 2>&1

# echo "random tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_random/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_random \
#     > "$SAVE_LOG_DIR/random_tuned.log" 2>&1

# (6)channel
# echo "channel"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/channel.log" 2>&1

# echo "channel tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_channel \
#     > "$SAVE_LOG_DIR/channel_tuned.log" 2>&1

# (7)Random-Layer
# echo "random 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.99 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cpu \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name layer \
#       --base_model $BASE_MODEL 

# echo "Random-Layer"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_random.log" 2>&1

# (8)PPL-Layer
# echo "imp 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.0 0.0 0.0 0.99 0.0 0.0 0.99 0.99 0.99 0.99 0.0 0.0 0.99 0.0 0.0 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cpu \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name layer \
#       --base_model $BASE_MODEL 

# echo "PPL-Layer"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_ppl.log" 2>&1


# (9)shortGPT
# echo "shortGPT 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.0 0.0 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cpu \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name layer \
#       --base_model $BASE_MODEL 

# echo "shortGPT"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/layer_shortgpt.log" 2>&1

# (10) our acc
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#     --pruning_ratio 0.0 0.31 0.0 0.0 0.17 0.0 0.0 0.0 0.0 0.17 0.0 0.0 0.2 0.31 0.08 0.49 0.29 0.0 0.29 0.18 0.41 0.45 0.32 0.46 0.32 0.48 0.37 0.0 0.0 0.47 0.42 0.0 \
#     --block_wise \
#     --pruner_type taylor \
#     --test_after_train \
#     --device cuda \
#     --eval_device cuda \
#     --save_model \
#     --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best \
#     --base_model $BASE_MODEL 

# echo "our"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best/pytorch_model.bin \
#     > "$SAVE_LOG_DIR/our_acc.log" 2>&1


# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python /home/iotsc01/xinpengq/LLM-Pruner-main/post_training.py \
#       --prune_model /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best/pytorch_model.bin \
#       --data_path yahma/alpaca-cleaned \
#       --lora_r 8 \
#       --num_epochs 2 \
#       --learning_rate 1e-4 \
#       --batch_size 64 \
#       --output_dir /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/acc_best \
#       --wandb_project llama_tune 

# echo "our tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=3 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/acc_best \
#     > "$SAVE_LOG_DIR/our_acc_tuned.log" 2>&1

# slicGPT
echo "sliceGPT"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
    --task_name bbh \
    --model_name llama \
    --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/TransformerCompression-main/llama \
    > "$SAVE_LOG_DIR/sliceGPT.log" 2>&1

# echo "SlicGPT tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/TransformerCompression-main/finetuned_model \
#     > "$SAVE_LOG_DIR/sliceGPT_tuned.log" 2>&1

# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name bbh \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/TransformerCompression-main/finetuned_model \
#     > "/home/iotsc01/xinpengq/LLM-Pruner-main/eval_logs_bbh/sliceGPT_tuned.log" 2>&1

