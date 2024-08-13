import pandas as pd
import numpy as np


PATHS = [
    './no_fine_tuning_gpt3/',
    './no_fine_tuning_gpt4/',
    './fine_tuning_gpt3/',
]

def combine_train_batches(path='.'):
    df1 = pd.read_json(path + 'train_result_1.jsonl', lines=True)
    df2 = pd.read_json(path + 'train_result_2.jsonl', lines=True)

    return pd.concat([df1, df2])

def fix_probability(label, probability):
    if label == '1':
        return probability
    else:
        return 1 - probability

def get_classifications(df, model_name='gpt', logprobs=True):
    # df = pd.read_json(path, lines=True)
    df = df.drop(columns=['id', 'error'])
    new_field_name = model_name + '_prediction'
    df = df.rename(columns={'response' : new_field_name})
    if logprobs:
        df[model_name + '_probability'] = df[new_field_name].apply(lambda x: np.exp(x['body']['choices'][0]['logprobs']['content'][0]['logprob']))
    df[new_field_name] = df[new_field_name].apply(lambda x: x['body']['choices'][0]['message']['content'])

    df[model_name + '_probability'] = df.apply(lambda row: fix_probability(row[new_field_name], row[model_name + '_probability']), axis=1)
    return df

training_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/train_set.csv'
validation_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/val_set.csv'
testing_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/test_set.csv'

train_set = pd.read_csv(training_path)
val_set = pd.read_csv(validation_path)

for path in PATHS:

    train_set = pd.merge(train_set, get_classifications(combine_train_batches(path), path.split('/')[1]), on='custom_id', how='inner')

    val_set = pd.merge(val_set, get_classifications(pd.read_json(path + f'val_result.jsonl', lines=True), path.split('/')[1]), on='custom_id', how='inner')


train_set = train_set.drop(columns=['text', 'translation'])
train_set.to_csv(f"./results/train_set_predictions.csv", encoding='utf-8', index=False, header=True)

val_set = val_set.drop(columns=['text', 'translation'])
val_set.to_csv(f"./results/validation_set_predictions.csv", encoding='utf-8', index=False, header=True)