import os
import time
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from create_classification_prompt_batches import GPT_3_MODEL, CLASSIFICATION_PROMPT

# Get API key from .env
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)


def create_example(message, answer):
    learning_example = {"messages": [
        {"role": "system", "content": CLASSIFICATION_PROMPT}, 
        {"role": "user", "content": message}, 
        {"role": "assistant", "content": str(answer)}
        ]}
    return learning_example


def create_files():
    train_file = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/train_set.csv'
    val_file = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/val_set.csv'
    train_file = pd.read_csv(train_file)
    val_file = pd.read_csv(val_file)

    train_save_file = f'./fine_tuning_gpt3/train_batch.jsonl'
    val_save_file = f'./fine_tuning_gpt3/val_batch.jsonl'

    output_file = open(train_save_file, "w")
    for i, row in train_file.iterrows():
        example = create_example(row['translation'], row['label'])
        output_file.write(json.dumps(example) + "\n")

    output_file = open(val_save_file, "w")
    for i, row in val_file.iterrows():
        example = create_example(row['translation'], row['label'])
        output_file.write(json.dumps(example) + "\n")

def fine_tune():
    train_save_file = f'./fine_tuning_gpt3/fine_tune_train_batch.jsonl'
    val_save_file = f'./fine_tuning_gpt3/fine_tune_val_batch.jsonl'

    train_file = client.files.create(
        file=open(train_save_file, "rb"),
        purpose="fine-tune"
    )
    val_file = client.files.create(
        file=open(val_save_file, "rb"),
        purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
        training_file=train_file.id, 
        validation_file=val_file.id,
        model=GPT_3_MODEL,
        suffix=f"sarcasm-train-val",
        hyperparameters={"n_epochs":3},
        seed=42
    )


if __name__ == "__main__":
    create_files()
    # fine_tune()