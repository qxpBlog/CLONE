#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=meta-llama/Llama-2-7b-hf
LOG_PATH=eval_logs

mkdir -p "$LOG_PATH"

# evaluate bbh
export PYTHONPATH='Pruner/lorahub1' 
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python hf_prune_bbh.py \
      --pruning_ratio 0.2 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name bbh \
      --base_model $BASE_MODEL \
      > "$LOG_PATH/bbh.log" 2>&1

# evaluate mmlu
export PYTHONPATH='.' 
CUDA_VISIBLE_DEVICES=0 python Pruner/LLaMA-Factory-main/src/evaluate.py \
      --model_name_or_path  Pruner/prune_log/llama_prune/pytorch_model.bin \
      --template vanilla \
      --finetuning_type lora \
      --task mmlu \
      --split validation \
      --lang en \
      --n_shot 5 \
      --batch_size 4 \
      > "$LOG_PATH/mmlu.log" 2>&1
    
# evaluate Commonsense
export PYTHONPATH='.' 
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python lm-evaluation-harness/main.py \
      --model hf-causal-experimental \
      --model_args checkpoint=Pruner/prune_log/llama_prune/pytorch_model.bin,config_pretrained=$BASE_MODEL \
      --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
      --no_cache \
      > "$LOG_PATH/Commonsense.log" 2>&1

# PPL
export PYTHONPATH='.' 
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=0 python hf_prune.py \
      --pruning_ratio 0.2 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name ppl \
      --base_model $BASE_MODEL \
      > "$LOG_PATH/ppl.log" 2>&1
