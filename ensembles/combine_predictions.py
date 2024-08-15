import pandas as pd
import os
import numpy as np

folder_path = './train_and_val_preds/meta-llama/'

def process_train_val(df, folder):
    if folder == '../small_models/results/' or folder == '../openai_api/results/':
        df = df[[x for x in df if x.endswith('_probability') or x.endswith('label') or x.endswith('custom_id')]]
    return df

def combine_train_val(folders):
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for folder in folders:
        temp_df = process_train_val(pd.read_csv(folder + 'train_set_predictions.csv'), folder)
        if train_df.empty:
            train_df = temp_df
        else:
            temp_df = temp_df.drop(columns=['label'])
            train_df = pd.merge(train_df, temp_df, on='custom_id', how='inner')

        temp_df = process_train_val(pd.read_csv(folder + 'validation_set_predictions.csv'), folder)
        if val_df.empty:
            val_df = temp_df
        else:
            temp_df = temp_df.drop(columns=['label'])
            val_df = pd.merge(val_df, temp_df, on='custom_id', how='inner')

    train_df.to_csv('./combined_model_predictions/train_set_predictions.csv', index=False)
    val_df = val_df[[x for x in train_df]]
    val_df.to_csv('./combined_model_predictions/validation_set_predictions.csv', index=False)
    print(train_df.head())



def fix_probability(label, probability):
    if label == 1 or label == '1':
        return probability
    else:
        return 1 - probability

def process_test(df, folder):
    if folder == '../openai_api/results/':
        logprobs_name = [x for x in df if x.endswith('_logprobs')][0]
        prediction_name = [x for x in df if x.endswith('_prediction')][0]

        df[logprobs_name[:-8] + 'probability'] = df.apply(lambda row: fix_probability(row[prediction_name], np.exp(row[logprobs_name])), axis=1)

    if folder == '../small_models/results/' or folder == '../openai_api/results/':
        df = df[[x for x in df if x.endswith('_probability') or x.endswith('label') or x.endswith('custom_id')]]

    return df

def combine_test(folders):
    test_df = pd.DataFrame()

    for folder in folders:
        for filename in os.listdir(folder):
            if not (filename.endswith('train_set_predictions.csv') or filename.endswith('validation_set_predictions.csv')):
                data_file = os.path.join(folder, filename)
                temp_df = process_test(pd.read_csv(data_file), folder)
                if test_df.empty:
                    test_df = temp_df
                else:
                    temp_df = temp_df.drop(columns=['label'])
                    test_df = pd.merge(test_df, temp_df, on='custom_id', how='inner')



    train_set = pd.read_csv('./combined_model_predictions/train_set_predictions.csv')
    test_df = test_df[[x for x in train_set]]
    test_df.to_csv('./combined_model_predictions/test_set_predictions.csv', index=False)

if __name__ == '__main__':
    folders =  [
        '../small_models/results/',
        '../large_models/results/meta-llama/',
        '../openai_api/results/',
        ]
    combine_train_val(folders)
    combine_test(folders)