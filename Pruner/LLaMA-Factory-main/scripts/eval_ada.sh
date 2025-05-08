#!/bin/bash

F1=/home/iotsc01/xinpengq/code-eval-main/eval_results/Adalora.log
F2=/home/iotsc01/xinpengq/code-eval-main/eval_results/IA3.log
F3=/home/iotsc01/xinpengq/code-eval-main/eval_results/P-Tuning.log
F4=/home/iotsc01/xinpengq/code-eval-main/eval_results/Prefix-tuning.log
F5=/home/iotsc01/xinpengq/code-eval-main/eval_results/Prompt_tuning.log

export WANDB_MODE=disabled
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/code-eval-main/eval_llama.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama/eval.jsonl > "$F1" 2>&1


