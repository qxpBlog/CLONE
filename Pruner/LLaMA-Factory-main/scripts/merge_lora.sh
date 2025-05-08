#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

export PYTHONPATH='/home/iotsc01/xinpengq/LLM-Pruner-main' 
CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/src/export_model.py \
    --model_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/pytorch_model.bin \
    --adapter_name_or_path /home/iotsc01/xinpengq/LLM-Pruner-main/tune_log/our \
    --template default \
    --finetuning_type lora \
    --export_dir /home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/our/our-merge \
    --export_size 2 \
    --export_legacy_format False