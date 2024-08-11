import os
import pandas as pd
from dotenv import load_dotenv

import torch
import transformers
from transformers.pipelines.pt_utils import KeyDataset


# Get API key from .env
load_dotenv()

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "./best_checkpoints/meta-llama/Meta-Llama-3.1-10B-Instruct/best_model",
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


def iterate_over_dataset(model_name):
    data = pd.read_csv("./train_val_test_split/test_set.csv")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16} if int(os.environ.get("MODEL_ID")) == 2 else None,
        device_map="auto",
        token=os.environ.get("ACCESS_TOKEN"),
    )

    data['prediction'] = data['translation'].apply(lambda text: ask_the_model(pipeline, CLASSIFICATION_PROMPT, text))
    data = data.drop(columns=['text', 'translation'])

    # data.to_csv(f'./results/{model_name}.csv', index=False) # For regular non-fine-tuned
    # data.to_csv(f'./results/meta-llama/Meta-Llama-3.1-70B-Instruct_fine_tuned.csv', index=False) # For fine-tuned

if __name__ == "__main__":
    iterate_over_dataset(MODELS[int(os.environ.get("MODEL_ID"))])
