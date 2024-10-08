import os
import sys
import argparse
import accelerate
from accelerate.utils import BnbQuantizationConfig
import torch
import numpy as np
import time
import transformers 
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoModel,AutoTokenizer,AutoModelForCausalLM,GPTQConfig
from codecarbon import track_emissions,EmissionsTracker
from LLMPruner.utils.logger import LoggerWithDepth
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from LLMPruner.evaluator.ppl import PPLMetric,test_latency_energy
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.peft import PeftModel
# from transformers import DistilBertTokenizer, DistilBertModel,BitsAndBytesConfig
from calflops import calculate_flops
# from huggingface_hub import login
 
# login()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def LlamaAttention_counter_hook(module, input, output):
    # (1) Ignore past-key values
    # (2) Assume there is no attention mask
    # Input will be empty in some pytorch version. use output here since input.shape == output.shape
    flops = 0
    q_len = output[0].shape[1]
    linear_dim = output[0].shape[-1]
    num_heads = module.num_heads
    head_dim = module.head_dim

    rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
    attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim) #QK^T + softmax + AttentionV
    linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim) # 4 for q, k, v, o. 
    flops += rotary_flops + attention_flops + linear_flops
    module.__flops__ += int(flops)

def rmsnorm_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    batch_flops *= 2
    module.__flops__ += int(batch_flops)

# @track_emissions()
def main(args):
    if args.test_mod == 'tuned':
        # 微调过后的模型的延迟和功耗的评估
        pruned_dict = torch.load(args.ckpt, map_location='cpu',)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        )
    elif args.test_mod == 'pruned':
        # 剪枝过后的模型的延迟和功耗的评估
        pruned_dict = torch.load(args.ckpt, map_location='cpu', )
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model.to('cuda') 
        model.half()
    elif args.test_mod == 'base':
        # tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        # model = LlamaForCausalLM.from_pretrained(
        #     args.base_model,
        #     low_cpu_mem_usage=True,
        #     torch_dtype=torch.float16
        # )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model,low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model.to('cuda')
    elif args.test_mod == 'quant':
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        
        # llm.int8
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map='cuda',
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    elif args.test_mod == 'distil':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model,low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model.to('cuda')  
    model.to('cuda')
    print(model.device)
    # model.config.pad_token_id = tokenizer.pad_token_id = 0 
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    model.eval()
    # 评估FLPOS和MACs
    # batch_size, max_seq_length = 1, 128
    # flops, macs, params = calculate_flops(model=model, 
    #                                   input_shape=(batch_size,max_seq_length),
    #                                   transformer_tokenizer=tokenizer)
    # print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    # 参数量、ppl、latency、energy
    after_pruning_parameters = sum(p.numel() for p in model.parameters())
    print("#parameters: {}".format(after_pruning_parameters))
    
    ppl = test_latency_energy(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device='cuda')
    print("PPL after pruning: {}".format(ppl))
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated() / 1024 / 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')
    
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b", help='base model name')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--test_mod', type=str, default="tuned", help='choose from [pruned, tuned, base]')
    args = parser.parse_args()

    main(args)
