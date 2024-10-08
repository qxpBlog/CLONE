MODEL=meta-llama/Llama-2-7b-hf
device=0
CUDA_VISIBLE_DEVICES=$device python main.py \
    --model_name_or_path $MODEL \
    --output_dir outputs/$MODEL



