from lorahub.algorithm import lorahub_inference,load_dataset
import os
import json
from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle
import torch
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig,AutoConfig



def evaluate_flan_results_zero_shot(folder, model, tokenizer):
    sub_dirs = os.listdir(folder)
    res = [0.0,0.0]
    model.eval()
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        
        # _, task_acc = lorahub_inference(task_inputs,
        #                   model,
        #                   tokenizer,
        #                   16,
        #                   task_outputs)
        def accuracy_score(outputs, ground_truths):
            correct = 0
            total = 0
            for output, truth in zip(outputs, ground_truths):
                # if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                if truth.strip().replace(".", "").replace("(", "").replace(")", "") in output.strip().replace(".", ""):
                    correct += 1
                total += 1
            return correct / total * 100

        example_predictions = []
           
        # process dataset
        dataset = load_dataset(task_inputs, task_outputs, tokenizer)
        # use gpu if available
        batch_size = 32
        for i in range(0, len(dataset["input"]), batch_size):
            inputs = tokenizer(
                dataset["input"][i : i + batch_size],
                max_length=2048,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                max_new_tokens=256
            )
            outputs = tokenizer.batch_decode(
                outputs.to("cpu"), skip_special_tokens=True
            )
            example_predictions.extend(outputs)
        for i in range(len(example_predictions)):
            index = example_predictions[i].find("A:")
            if index != -1:
                example_predictions[i] = example_predictions[i][index+2:]
        if task_outputs is not None:
            task_perf = accuracy_score(example_predictions, task_outputs)
        else:
            task_perf = None
        print("***",task_perf,"***")
        res[0] += 1
        res[1] += float(task_perf)
        torch.cuda.empty_cache()
    print("result avg:",res[1]/res[0])
        

def evaluate_flan_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    res = [0,0]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(flan_model_name)
    tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
    
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"): 
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (few shot): ", sub_dir)
        
        # _, task_acc = lorahub_inference(task_inputs,
        #                   model,
        #                   tokenizer,
        #                   16,
        #                   task_outputs)
        def accuracy_score(outputs, ground_truths):
            correct = 0
            total = 0
            for output, truth in zip(outputs, ground_truths):
                # if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                if truth.strip().replace(".", "") in output.strip().replace(".", ""):
                    correct += 1
                total += 1
            return correct / total * 100

        example_predictions = []
           
        # process dataset
        dataset = load_dataset(task_inputs, task_outputs, tokenizer)
        # use gpu if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        for i in range(0, len(dataset["input"]), 2):
            inputs = tokenizer(
                dataset["input"][i : i + 2],
                max_length=2048,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            outputs = model.generate(
                input_ids=inputs["input_ids"], max_new_tokens=5
            )
            outputs = tokenizer.batch_decode(
                outputs.to("cpu"), skip_special_tokens=True
            )
            example_predictions.extend(outputs)
        for i in range(len(example_predictions)):
            index = example_predictions[i].rfind("A:")
            if index != -1:
                example_predictions[i] = example_predictions[i][index+2:]
            
        if task_outputs is not None:
            task_perf = accuracy_score(example_predictions, task_outputs)
        else:
            task_perf = None
        print("***",task_perf,"***")
        res[0]+=1
        res[1]+=task_perf
        torch.cuda.empty_cache()
    print("result avg:",res[1]/res[0])


def evaluate_lorahub_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)

    # 5 seeds used in our experiments
    for sub_dir in sub_dirs:
        # construct the few-shot examples for lorahub learning
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            
        # random select 5 examples for each task
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        # load the zero-shot examples for evaluation
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])

        task_perf_list = []
        for seed in range(1, 6):
            random.seed(seed)

            def get_lora_module_list():
                return random.sample(LORA_MODULE_NAMES, 20)
            # get a list of modules to be used in the composition
            modules = get_lora_module_list()

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                                example_inputs=example_inputs,
                                                                example_outputs=examples_outputs,
                                                                max_inference_step=40,
                                                                batch_size=5)

            print("module_weights:", module_weights)

            """
            Perform inference to get predictions
            """
            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=10,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=task_outputs)
            task_perf_list.append(task_acc)
        avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        print("average perf:", avg_perf, "best perf:", max_perf)


if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")
    # evaluate the model
    # evaluate_flan_results_zero_shot("data_bbh", "google/flan-t5-large")
    # evaluate_flan_results_zero_shot("data_bbh", "/home/iotsc01/xinpengq/LLM-Pruner-main/llama-7b-hf")
    evaluate_flan_results_zero_shot("data_bbh", "/home/iotsc01/xinpengq/LLM-Pruner-main/prune_log/llama_prune_our/pytorch_model.bin")
    # # five shot for flan models
    # evaluate_flan_results_few_shot("data_bbh", "google/flan-t5-large")
    # evaluate_flan_results_few_shot("data_bbh", "/home/iotsc01/zsh/model/gemma-7b")
    # # five shot for lorahub models
    # evaluate_lorahub_results_few_shot("data_bbh", "google/flan-t5-large")
