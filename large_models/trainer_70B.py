import numpy as np
import pandas as pd
import json
import torch
import random
import optuna
import torch.optim as optim
import os
from dotenv import load_dotenv
from datasets import load_dataset
from preprocess_examples import CLASSIFICATION_PROMPT

import evaluate
from transformers import set_seed
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model


# Get API key from .env
load_dotenv()

set_seed(42)
random.seed(10)
np.random.seed(42)
torch.manual_seed(42)


def prepare_data(path_to_data='.', train_val='train'):
    dataset = load_dataset("json", data_files=path_to_data, split=train_val)
    return dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit 
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization 
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def run_training(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.environ.get("ACCESS_TOKEN"),)
    tokenizer.pad_token = tokenizer.eos_token

    training_path = "./train_val_test_split_preprocessed/train_set.jsonl"
    validation_path = "./train_val_test_split_preprocessed/val_set.jsonl"
    testing_path = "./train_val_test_split_preprocessed/test_set.jsonl"

    train_set = prepare_data(training_path, 'train')
    val_set = prepare_data(validation_path, 'train')
    test_set = prepare_data(testing_path, 'train')

    STEPS = 80

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    training_args = TrainingArguments(
        save_strategy="steps",
        save_steps=STEPS,
        save_total_limit=1,
        output_dir=f"./best_checkpoints/{model_name}/",
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=STEPS,
        num_train_epochs=10,
        log_level="info",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        learning_rate=1e-5,
        weight_decay=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        optim='paged_adamw_32bit', #specialization of the AdamW optimizer that enables efficient learning in LoRA setting.
        lr_scheduler_type="constant",
        seed=42,
        data_seed=42,
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        token=os.environ.get("ACCESS_TOKEN"),
        trust_remote_code=True
    )
    model.max_new_tokens = 1

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=val_set,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=256,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(f"./best_checkpoints/{model_name}/best_model")


if __name__ == "__main__":
    MODELS = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ]

    run_training(MODELS[int(os.environ.get("MODEL_ID"))])