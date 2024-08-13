import os
import time
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from openai_request import create_request

# Get environment variables from .env
load_dotenv()

MODEL = "GPT-4o-2024-05-13"

CLASSIFICATION_PROMPT = '''You will be provided with text in the Slovenian language, and your task is to classify whether it is sarcastic or not. Use ONLY token 0 (not sarcastic) or 1 (sarcastic) as in the examples:
Spanje? Kaj je to... Še nikoli nisem slišal za to? 1
Lepo je biti primerjan z zidom 😂 1
To sploh nima smisla. Nehaj kopati. 0
Dne 12. 10. 21 ob 10:30 je bil nivo reke 0,37 m. 0'''

def prepare_original_data():
    # Combine train and test
    train = pd.read_csv('../datasets/translatediSarcasmEval_T5/train.csv')
    train = train[['tweet', 'sarcastic']]
    train = train.rename(columns={'tweet': 'text', 'sarcastic': 'label'})

    test = pd.read_csv('../datasets/translatediSarcasmEval_T5/test_A.csv')
    test = test[['text', 'sarcastic']]
    test = test.rename(columns={'text': 'text', 'sarcastic': 'label'})

    # Ensure balance
    dataset = pd.concat([train, test], ignore_index=True)
    df_1 = dataset[dataset['label'] == 1]
    df_0 = dataset[dataset['label'] == 0]

    min_count = min(len(df_1), len(df_0))
    
    # Sample the smaller count from both dataframes
    df_1_sampled = df_1.sample(n=min_count, random_state=1)
    df_0_sampled = df_0.sample(n=min_count, random_state=1)

    dataset = pd.concat([df_1_sampled, df_0_sampled], ignore_index=True) 
    return dataset


def create_batch(dataset, folder='translate', text_field='text'):
    # Create batch files that fit the API rate limit
    for i, row in dataset.iterrows():
        if i % 500 == 0:
            batch_file = open(f"./{folder}/batch_{i}.jsonl", "w")
        identifier = f"translation-{i}"
        dataset.loc[i, "identifier"] = identifier
        request = create_request(identifier=identifier, model=MODEL, system_promt=TRANSLATION_PROMPT, user_message=row[text_field], logprobs=False)
        batch_file.write(json.dumps(request) + "\n")

def send_batch(folder='translate'):
    batch_identifiers = []
    # Send all batch files
    for index in [0, 500, 1000, 1500, 2000]:
        print("Sending batch:", index)
        batch_input_file = client.files.create(
            file=open(f"./{folder}/batch_{index}.jsonl", "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        metadata = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{folder} iSarcasmEval"
            }
        )
        # Save batch id and wait for batch to finish
        batch_identifiers.append(metadata.id)
        while client.batches.retrieve(metadata.id).status != "completed":
            print("\rWaiting for completion...", end="")
            # Wait 3 minutes for batch to complete
            time.sleep(60)
        print("\rFinished!                           ") # Delete waiting message

    return batch_identifiers

def retrieve_batches(batch_identifiers, folder='translate'):
    for i, identifier in enumerate(batch_identifiers):
        batch_job = client.batches.retrieve(identifier)
        file_response = client.files.content(batch_job.output_file_id).content
        result_file = open(f"./{folder}/batch_result_{i}.jsonl", "wb")
        result_file.write(file_response)

if __name__ == '__main__':
    dataset = prepare_original_data()
    create_batch(dataset, 'translate_gpt4', 'text')
    batch_identifiers = send_batch('translate_gpt4')
    retrieve_batches(batch_identifiers, 'translate_gpt4')