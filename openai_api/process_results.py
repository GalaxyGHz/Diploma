import pandas as pd
import json


original_dataset = pd.read_csv('../datasets/translatediSarcasmEval_ChatGPT4o/dataset.csv')

def combine_translations(folder='translate', name='translation', logprobs=False, save_path='.'):
    results = []
    for i in range(5):
        df = pd.read_json(f'./{folder}/batch_result_{i}.jsonl', lines=True)
        df = df.drop(columns=['id', 'error'])
        df = df.rename(columns={'response' : name})
        if logprobs:
            df['logprobs'] = df[name].apply(lambda x: x['body']['choices'][0]['logprobs']['content'][0]['logprob'])
        df[name] = df[name].apply(lambda x: x['body']['choices'][0]['message']['content'])
        results.append(df)

    result = pd.concat(results, axis=0)

    dataset = pd.merge(original_dataset, result, on='custom_id', how='inner')
    dataset = dataset.drop(columns=['text'])
    print(len(dataset))
    print(dataset.head())

    dataset.to_csv(save_path, encoding='utf-8', index=False, header=True)

def get_classifications(data='.', new_field_name='prediction', logprobs=False, save_path='.'):
    df = pd.read_json(data, lines=True)
    df = df.drop(columns=['id', 'error'])
    df = df.rename(columns={'response' : new_field_name})
    if logprobs:
        df['logprobs'] = df[new_field_name].apply(lambda x: x['body']['choices'][0]['logprobs']['content'][0]['logprob'])
    df[new_field_name] = df[new_field_name].apply(lambda x: x['body']['choices'][0]['message']['content'])

    dataset = pd.merge(original_dataset, df, on='custom_id', how='inner')
    dataset = dataset.drop(columns=['text'])
    print(len(dataset))
    print(dataset.head())

    dataset.to_csv(save_path, encoding='utf-8', index=False, header=True)

if __name__ == '__main__':
    # combine_translations('translate', 'translation', False, '../datasets/translatedSarcasmEval_ChatGPT4o/translated_dataset.csv')

    # get_classifications('./no_fine_tuning_gpt3/test_result.jsonl', 'prediction', True, './results/gpt3_no_fine_tuning.csv')
    # get_classifications('./no_fine_tuning_gpt4/test_result.jsonl', 'prediction', True, './results/gpt4_no_fine_tuning.csv')
    get_classifications('./fine_tuning_gpt3/test_result.jsonl', 'prediction', True, './results/gpt3_fine_tuning.csv')