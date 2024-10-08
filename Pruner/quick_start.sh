#!/bin/bash
HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL=meta-llama/Llama-2-7b
ORTHER_PRUNING_PATH=LLM-Pruner-main/eval_logs

mkdir -p "$ORTHER_PRUNING_PATH"

echo "20% block pruning"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python hf_prune.py \
      --pruning_ratio 0.2 \
      --block_wise \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_model \
      --save_ckpt_log_name llama_prune_0.2 \
      --base_model $BASE_MODEL \
      > "$ORTHER_PRUNING_PATH/test_latency_energy_0.2.log" 2>&1


export PYTHONPATH='.' 
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python lm-evaluation-harness/main.py \
      --model hf-causal-experimental \
      --model_args checkpoint=prune_log/llama_prune_0.2/pytorch_model.bin,config_pretrained=$BASE_MODEL \
      --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
      --no_cache \
      > "$ORTHER_PRUNING_PATH/0.2_mlt.log" 2>&1

echo "20% block fine-tuning"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python post_training.py \
      --prune_model prune_log/llama_prune_0.2/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir tune_log_llama2/0.2 \
      --wandb_project 0.2 

echo "fine-tuninged 20% block test latecy and energy"
HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python test_latency_energy.py \
      --ckpt prune_log/llama_prune_0.2/pytorch_model.bin \
      --lora_ckpt tune_log_llama2/0.2 \
      > "$ORTHER_PRUNING_PATH/fine-tuning_0.2_test_latency_energy.log" 2>&1


export PYTHONPATH='.' 
    HF_ENDPOINT=$HF_ENDPOINT CUDA_VISIBLE_DEVICES=1 python lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args checkpoint=prune_log/llama_prune_0.2/pytorch_model.bin,peft=tune_log_llama2/0.2,config_pretrained=$BASE_MODEL \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --no_cache \
        > "$ORTHER_PRUNING_PATH/fine-tuning_0.2_test_mlt.log" 2>&1