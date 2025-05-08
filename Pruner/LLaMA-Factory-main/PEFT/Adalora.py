from functools import partial
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from peft import (get_peft_config, get_peft_model, LoraConfig, TaskType, IA3Model, IA3Config, 
                  PromptEncoder, PromptEncoderConfig, get_peft_model_state_dict,
                set_peft_model_state_dict, PrefixTuningConfig,
                PeftType, PromptTuningInit, PromptTuningConfig,
                TaskType,AdaLoraModel, AdaLoraConfig,)

from transformers import TrainingArguments, set_seed
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb
import os
import torch

# Reproducibility
seed = 42
set_seed(seed)
# ============================load model ===============================
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))


peft_config = AdaLoraConfig(
    peft_type="ADALORA", task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    lora_dropout=0.05,
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()



# ==========================preparing dataset =================================
task_name='code_alpaca_20k'
dataset_prompt = "/home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/data/{}.json".format(task_name)
def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["input"] = formatted_prompt

    return sample
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["input"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


#Create the Dataset to create prompts.
data_prompt = load_dataset('json', data_files = dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["instruction"]), batched=True)
train_sample_prompt = data_prompt["train"]

# max_length = get_max_length(model)
# dataset = load_dataset('json', data_files = dataset_prompt)
# train_sample_prompt = preprocess_dataset(tokenizer, max_length, seed, dataset)["train"]




#================================== save path ===================================
#Just creating the directoris if not exist.
working_dir = "/home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint_adalora"
output_directory_prompt =  os.path.join(working_dir, "llama2_code")
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(output_directory_prompt):
    os.mkdir(output_directory_prompt)
    



# ==========================training ==================================
def create_training_arguments(path, learning_rate=5e-5, epochs=3):
    training_args = TrainingArguments(
        output_dir=path, # Where the model predictions and checkpoints will be written
        use_cpu=False, # This is necessary for CPU clusters.
        per_device_train_batch_size=4, 
        gradient_accumulation_steps =4,
        learning_rate= learning_rate, # Higher learning rate than full fine-tuning
        num_train_epochs=epochs,
        logging_steps = 10,
        save_steps = 1000,
        fp16=True,
        optim="paged_adamw_8bit",
    )
    return training_args

NUM_EPOCHS = 3
training_args_prompt = create_training_arguments(output_directory_prompt, 5e-5, NUM_EPOCHS)


def create_trainer(model, training_args, train_dataset):
    trainer = Trainer(
        model=model, # We pass in the PEFT version of the foundation model, bloomz-560M
        args=training_args, #The args for the training.
        train_dataset=train_dataset, #The dataset used to tyrain the model.
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
    )
    return trainer

#Training  model.
trainer_prompt = create_trainer(model, training_args_prompt, train_sample_prompt)
trainer_prompt.train()
model.save_pretrained(output_directory_prompt)
tokenizer.save_pretrained(output_directory_prompt)
