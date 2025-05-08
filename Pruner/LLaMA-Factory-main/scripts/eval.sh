#!/bin/bash

F1=/home/iotsc01/xinpengq/code-eval-main/eval_results/Adalora.log
F2=/home/iotsc01/xinpengq/code-eval-main/eval_results/IA3.log
F3=/home/iotsc01/xinpengq/code-eval-main/eval_results/P-Tuning.log
F4=/home/iotsc01/xinpengq/code-eval-main/eval_results/Prefix-tuning.log
F5=/home/iotsc01/xinpengq/code-eval-main/eval_results/Prompt_tuning.log

export WANDB_MODE=disabled

# CUDA_VISIBLE_DEVICES=1,2 accelerate launch /home/iotsc01/xinpengq/code-eval-main/eval_llama.py
AdaLoRa
echo "AdaLoRa微调开始"
CUDA_VISIBLE_DEVICES=1 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/PEFT/Adalora.py

echo "AdaLoRa微调结束"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/code-eval-main/eval_llama_adalora.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_adalora/eval.jsonl > "$F1" 2>&1
# evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_adalora/eval.jsonl > "/home/iotsc01/xinpengq/code-eval-main/eval_results/Adalora.log" 2>&1
# rm -r /home/iotsc01/xinpengq/code-eval-main/results
# rm -r /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint

# IA3
echo "IA3微调开始"
CUDA_VISIBLE_DEVICES=1 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/PEFT/IA3.py
echo "IA3微调结束"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python /home/iotsc01/xinpengq/code-eval-main/eval_llama_IA3.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_IA3/eval.jsonl > "$F2" 2>&1
# evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_IA3/eval.jsonl > "/home/iotsc01/xinpengq/code-eval-main/eval_results/IA3.log" 2>&1
# rm -r /home/iotsc01/xinpengq/code-eval-main/results
# rm -r /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint

# P-Tuning
echo "P-Tuning微调开始"
CUDA_VISIBLE_DEVICES=2 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/PEFT/P-Tuning.py
echo "P-Tuning微调结束"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/code-eval-main/eval_llama_p_tuning.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_p-tuning/eval.jsonl > "$F3" 2>&1
# evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_p-tuning/eval.jsonl > "/home/iotsc01/xinpengq/code-eval-main/eval_results/P-Tuning.log" 2>&1
# rm -r /home/iotsc01/xinpengq/code-eval-main/results
# rm -r /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint

# perfix-tuning
echo "perfix-tuning微调开始"
CUDA_VISIBLE_DEVICES=3 python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/PEFT/Prefix-tuning.py
echo "perfix-tuning微调结束"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 /home/iotsc01/anaconda3/envs/xinpengq_env/bin/python /home/iotsc01/xinpengq/code-eval-main/eval_llama_prefix_tuning.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_prefix_tuning/eval.jsonl > "$F4" 2>&1
# evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_prefix_tuning/eval.jsonl > "/home/iotsc01/xinpengq/code-eval-main/eval_results/Prefix-tuning.log" 2>&1
# rm -r /home/iotsc01/xinpengq/code-eval-main/results
# rm -r /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint


# prompt-tuning
echo "prompt微调开始"
CUDA_VISIBLE_DEVICES=3 python /home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/PEFT/Prompt_tuning.py
echo "prompt微调结束"

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 python /home/iotsc01/xinpengq/code-eval-main/eval_llama_prompt_tuning.py

evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_prompt_tuning/eval.jsonl > "$F5" 2>&1
# evaluate_functional_correctness /home/iotsc01/xinpengq/code-eval-main/results/llama_prompt_tuning/eval.jsonl > "/home/iotsc01/xinpengq/code-eval-main/eval_results/Prompt_tuning.log" 2>&1