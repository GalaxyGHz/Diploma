import json
import pandas as pd
from openai_request import create_request

GPT_3_MODEL = "gpt-3.5-turbo-0125"
GPT_3_MODEL_FINE_TUNED = "ft:gpt-3.5-turbo-0125:personal:sarcasm-train-val:9uICh2ML"
GPT_4_MODEL = "gpt-4o"  # at time of writting points to GPT-4o-2024-05-13

CLASSIFICATION_PROMPT = '''You will be provided with text in the Slovenian language, and your task is to classify whether it is sarcastic or not. Use ONLY token 0 (not sarcastic) or 1 (sarcastic) as in the examples:
    Spanje? Kaj je to... Še nikoli nisem slišal za to? 1
    Lepo je biti primerjan z zidom 😂 1
    To sploh nima smisla. Nehaj kopati. 0
    Dne 12. 10. 21 ob 10:30 je bil nivo reke 0,37 m. 0'''

def create_batches_no_fine_tuning(folder, model=GPT_4_MODEL):
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/{split}_set.csv')
        print(df.head())

        if split != 'train':
            batch_file = open(folder + split + '_batch.jsonl', "w")
            for j, row in df.iterrows():
                request = create_request(identifier=row['custom_id'], model=model, system_promt=CLASSIFICATION_PROMPT, user_message=row['translation'], logprobs=True)
                batch_file.write(json.dumps(request) + "\n")
        else:
            batch_file_1 = open(folder + split + '_batch_1.jsonl', "w")
            batch_file_2 = open(folder + split + '_batch_2.jsonl', "w")
            for j, row in df.iterrows():
                request = create_request(identifier=row['custom_id'], model=model, system_promt=CLASSIFICATION_PROMPT, user_message=row['translation'], logprobs=True)
                if j < 640: # Split in two since there is a 90k token limit on batches
                    batch_file_1.write(json.dumps(request) + "\n")
                else:
                    batch_file_2.write(json.dumps(request) + "\n")

        
if __name__ == '__main__':
    create_batches_no_fine_tuning("./no_fine_tuning_gpt4/", GPT_4_MODEL)
    create_batches_no_fine_tuning("./no_fine_tuning_gpt3/", GPT_3_MODEL)
    create_batches_no_fine_tuning("./fine_tuning_gpt3/", GPT_3_MODEL_FINE_TUNED)
