import datasets
from datasets import load_dataset
import os
import pandas as pd
import json

CLASSIFICATION_PROMPT = '''You will be provided with text in the Slovenian language, and your task is to classify whether it is sarcastic or not. Use ONLY token 0 (not sarcastic) or 1 (sarcastic) as in the examples:
    Spanje? Kaj je to... Še nikoli nisem slišal za to? 1
    Lepo je biti primerjan z zidom 😂 1
    To sploh nima smisla. Nehaj kopati. 0
    Dne 12. 10. 21 ob 10:30 je bil nivo reke 0,37 m. 0'''


def create_example(message, answer):
    learning_example = {"messages": [
        {"role": "system", "content": CLASSIFICATION_PROMPT}, 
        {"role": "user", "content": message}, 
        {"role": "assistant", "content": str(answer)}
        ]}
    return learning_example

def go_over_split(file_name):
    original_directory = "./train_val_test_split"
    new_directory = "./train_val_test_split_preprocessed"

    data = pd.read_csv(os.path.join(original_directory, file_name))

    output_file = open(os.path.join(new_directory, file_name.split('.')[0] + '.jsonl'), 'w')
    for i, row in data.iterrows():
        example = create_example(row['translation'], str(row['label']))
        output_file.write(json.dumps(example) + "\n")


def preprocess_examples():
    directory = "./train_val_test_split"
    for file_name in os.listdir(directory):
        go_over_split(file_name)



if __name__ == "__main__":
    preprocess_examples()

    dataset = load_dataset("json", data_files="./train_val_test_split_preprocessed/train_set.jsonl", split="train")
    print(dataset)