from transformers import AutoModelForCausalLM, LlamaForCausalLM
from peft import (get_peft_config, get_peft_model, LoraConfig, TaskType, IA3Model, IA3Config, 
                  PromptEncoder, PromptEncoderConfig, get_peft_model_state_dict,
                set_peft_model_state_dict, PrefixTuningConfig,
                PeftType, PromptTuningInit, PromptTuningConfig,
                TaskType,AdaLoraModel, AdaLoraConfig,)

from transformers import TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from datasets import load_dataset
import os


# ============================load model ===============================
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))


peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()



# ==========================preparing dataset =================================
task_name='code_alpaca_20k'
dataset_prompt = "/home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/data/{}.json".format(task_name)

#Create the Dataset to create prompts.
data_prompt = load_dataset('json', data_files = dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["instruction"]), batched=True)
train_sample_prompt = data_prompt["train"]





#================================== save path ===================================
#Just creating the directoris if not exist.
working_dir = "/home/iotsc01/xinpengq/LLM-Pruner-main/LLaMA-Factory-main/sft_checkpoint_prefix_tuning"
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
        fp16=True
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