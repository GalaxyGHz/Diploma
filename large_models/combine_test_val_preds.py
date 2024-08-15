import pandas as pd
import os


folder_path = './train_and_val_preds/meta-llama/'

train_df = pd.DataFrame()
val_df = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('train_set.csv'):
        data_file = os.path.join(folder_path, filename)
        temp_df = pd.read_csv(data_file)
        if train_df.empty:
            train_df = temp_df
        else:
            temp_df = temp_df.drop(columns=['label'])
            train_df = pd.merge(train_df, temp_df, on='custom_id', how='inner')

    elif filename.endswith('val_set.csv'):
        data_file = os.path.join(folder_path, filename)
        temp_df = pd.read_csv(data_file)
        temp_df = pd.read_csv(os.path.join(folder_path, filename))
        if val_df.empty:
            val_df = temp_df
        else:
            temp_df = temp_df.drop(columns=['label'])
            val_df = pd.merge(val_df, temp_df, on='custom_id', how='inner')

train_df.to_csv('./results/meta-llama/train_set_predictions.csv', index=False)
val_df.to_csv('./results/meta-llama/validation_set_predictions.csv', index=False)

print(train_df.head())