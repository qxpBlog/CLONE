#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf
SAVE_PATH=/home/iotsc01/xinpengq/LLM-Pruner-main/eval_mmlu

mkdir -p "$SAVE_PATH"

# (1)base model
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#     --pruning_ratio 0.0 \
#     --block_wise \
#     --pruner_type taylor \
#     --test_after_train \
#     --device cuda \
#     --eval_device cuda \
#     --save_model \
#     --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base \
#     --base_model $BASE_MODEL 

# echo "base"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base/pytorch_model.bin \
#     > "$SAVE_PATH/base.log" 2>&1

# echo "base tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/base_model \
#     > "$SAVE_PATH/base_tuned.log" 2>&1

# # (2)our

# 0.0 0.31 0.0 0.0 0.17 0.0 0.0 0.0 0.0 0.17 0.0 0.0 0.2 0.31 0.08 0.49 0.29 0.0 0.29 0.18 0.41 0.45 0.32 0.46 0.32 0.48 0.37 0.0 0.0 0.47 0.42 0.0
# 0.0 0.0 0.0 0.0 0.0 0.704 0.0 0.0 0.697 0.0 0.0 0.513 0.0 0.0 0.0 0.647 0.0 0.911 0.0 0.745 0.31 0.0 0.0 0.0 0.587 0.0 0.0 0.497 0.637 0.0 0.0 0.0
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#     --pruning_ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.61 0.0 0.0 0.0 0.2 0.0 0.57 0.0 0.09 0.0 0.1 0.0 0.59 0.45 0.63 0.46 0.63 0.48 0.56 0.0 0.0 0.89 0.0 0.0 \
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
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best/pytorch_model.bin \
#     > "$SAVE_PATH/our_acc.log" 2>&1

# echo "our tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/acc_best/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/eurosys_our_acc_0 \
#     > "$SAVE_PATH/our_acc_tuned.log" 2>&1

# # (3)20%
# echo "20% pruning"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.2 \
#       --block_wise \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base \
#       --base_model $BASE_MODEL 

# echo "20%"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base/pytorch_model.bin \
#     > "$SAVE_PATH/0.2.log" 2>&1

# echo "20% tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2 \
#     > "$SAVE_PATH/0.2_tuned.log" 2>&1

# # (4)L2
# echo "l2"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_l2/pytorch_model.bin \
#     > "$SAVE_PATH/l2.log" 2>&1

# echo "l2 tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_l2/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_l2 \
#     > "$SAVE_PATH/l2_tuned.log" 2>&1

# # (5)Random
# echo "random pruning"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
#       --pruning_ratio 0.2 \
#       --block_wise \
#       --pruner_type random \
#       --test_after_train \
#       --device cuda \
#       --eval_device cuda \
#       --save_model \
#       --save_ckpt_log_name /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base \
#       --base_model $BASE_MODEL 

# echo "random"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base \
#     > "$SAVE_PATH/random.log" 2>&1

# echo "random tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/base \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_random \
#     > "$SAVE_PATH/random_tuned.log" 2>&1

# # (6)channel
# echo "channel"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
#     > "$SAVE_PATH/channel.log" 2>&1

# echo "channel tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_0.2_channel/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/0.2_channel \
#     > "$SAVE_PATH/channel_tuned.log" 2>&1

# # (7)Random-Layer
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
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_PATH/layer_random.log" 2>&1

# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/random_layer \
#     > "$SAVE_PATH/layer_random_tuned.log" 2>&1

# # (8)PPL-Layer
# echo "imp 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
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
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_PATH/layer_ppl.log" 2>&1


# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/ppl_layer \
#     > "$SAVE_PATH/layer_ppl_tuned.log" 2>&1

# (9)shortGPT
# echo "shortGPT 7 layers"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/hf_prune.py \
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
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     > "$SAVE_PATH/layer_shortgpt.log" 2>&1

# echo "shotGPT layer tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path  /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/layer/pytorch_model.bin \
#     --lora_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/shortgpt \
#     > "$SAVE_PATH/layer_shotGPT_tuned.log" 2>&1

# (10) slicGPT
# echo "sliceGPT"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/TransformerCompression-main/llama \
#     > "$SAVE_PATH/sliceGPT.log" 2>&1

# echo "SlicGPT tuned"
# export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
# HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task_name mmlu \
#     --model_name llama \
#     --model_path /home/iotsc01/xinpengq/LLM-Pruner-main/TransformerCompression-main/finetuned_model \
#     > "$SAVE_PATH/sliceGPT_tuned.log" 2>&1

# (11)LoRAMOE
echo "LoRAMOE"
export PYTHONPATH="/home/iotsc01/xinpengq/LLM-Pruner-main"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=2 python main.py \
    --task_name mmlu \
    --model_name llama \
    --model_path $BASE_MODEL \
    --lora_path /home/iotsc01/xinpengq/LoRAMoE-main/output/dolly/sft_lora_model \
    > "$SAVE_PATH/loramoe.log" 2>&1


