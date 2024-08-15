import os
import pandas as pd
from dotenv import load_dotenv

import torch
import transformers
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model


# Get API key from .env
load_dotenv()

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "./best_checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/best_model",
    "./best_checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct/best_model",
]

CLASSIFICATION_PROMPT = '''You will be provided with text in the Slovenian language, and your task is to classify whether it is sarcastic or not. Use ONLY token 0 (not sarcastic) or 1 (sarcastic) as in the examples:
    Spanje? Kaj je to... Še nikoli nisem slišal za to? 1
    Lepo je biti primerjan z zidom 😂 1
    To sploh nima smisla. Nehaj kopati. 0
    Dne 12. 10. 21 ob 10:30 je bil nivo reke 0,37 m. 0'''

def ask_the_model(pipeline, system_prompt, user_message):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1,
    )
    return outputs[0]["generated_text"][-1]['content']

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit 
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization 
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def classify_and_save(pipeline, data, save_path, model_name):
    data[model_name + '_prediction'] = data['translation'].apply(lambda text: ask_the_model(pipeline, CLASSIFICATION_PROMPT, text))
    data = data.drop(columns=['text', 'translation'])

    data.to_csv(save_path, index=False) 

def iterate_over_dataset(model_name):
    test_set = pd.read_csv("./train_val_test_split/test_set.csv")
    train_set = pd.read_csv("./train_val_test_split/train_set.csv")
    val_set = pd.read_csv("./train_val_test_split/val_set.csv")

    model = model_name

    if int(os.environ.get("MODEL_ID")) == 3:
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3.1-8B-Instruct', 
            token=os.environ.get("ACCESS_TOKEN"),
            trust_remote_code=True
        )
        lora_config = LoraConfig.from_pretrained(model_name)
        model = get_peft_model(model, lora_config)

    elif int(os.environ.get("MODEL_ID")) == 4:
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3.1-70B-Instruct',
            quantization_config=bnb_config,
            token=os.environ.get("ACCESS_TOKEN"),
            trust_remote_code=True
        )
        model.max_new_tokens = 1
        lora_config = LoraConfig.from_pretrained(model_name)
        model = get_peft_model(model, lora_config)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model, 
        tokenizer=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16} if int(os.environ.get("MODEL_ID")) == 2 else None,
        # device_map="auto", # Not fine-tuned
        device='cuda', # for 8B fine-tuned model
        token=os.environ.get("ACCESS_TOKEN"),
    )

    # For regular non-fine-tuned
    # classify_and_save(pipeline, test_set, f'./results/{model_name}.csv', model_name.split('/')[1])
    # classify_and_save(pipeline, val_set, f'./train_and_val_preds/{model_name}_val_set.csv', model_name.split('/')[1])
    # classify_and_save(pipeline, train_set, f'./train_and_val_preds/{model_name}_train_set.csv', model_name.split('/')[1])

    # For fine-tuned
    model_size = 8
    classify_and_save(pipeline, test_set, f'./results/meta-llama/Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned.csv', f'Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned')
    classify_and_save(pipeline, train_set, f'./train_and_val_preds/meta-llama/Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned_train_set.csv', f'Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned')
    classify_and_save(pipeline, val_set, f'./train_and_val_preds/meta-llama/Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned_val_set.csv', f'Meta-Llama-3.1-{model_size}B-Instruct_fine_tuned')


if __name__ == "__main__":
    iterate_over_dataset(MODELS[int(os.environ.get("MODEL_ID"))])
