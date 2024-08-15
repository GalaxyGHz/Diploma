import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def calculate_accuracy_and_f1(path_to_preds='.'):
    predictions = pd.read_csv(path_to_preds)

    label = predictions['label']
    preds = predictions[path_to_preds.split('/')[-1][:-4] + '_prediction']

    # preds = predictions[path_to_preds.split('/')[-1][:-14] + '_prediction']

    f1 = f1_score(y_true=label, y_pred=preds)
    acccuracy = accuracy_score(y_true=label, y_pred=preds)
    print(path_to_preds, ":", "accuracy:", acccuracy, "f1:", f1)



if __name__ == '__main__':
    calculate_accuracy_and_f1('./results/meta-llama/Meta-Llama-3.1-8B-Instruct.csv')
    calculate_accuracy_and_f1('./results/meta-llama/Meta-Llama-3.1-8B-Instruct_fine_tuned.csv')

    calculate_accuracy_and_f1('./results/meta-llama/Meta-Llama-3.1-70B-Instruct.csv')
    calculate_accuracy_and_f1('./results/meta-llama/Meta-Llama-3.1-70B-Instruct_fine_tuned.csv')

    calculate_accuracy_and_f1('./results/meta-llama/Meta-Llama-3.1-405B-Instruct.csv')

    # calculate_accuracy_and_f1('./train_and_val_preds/meta-llama/Meta-Llama-3.1-8B-Instruct_fine_tuned_train_set.csv')
    # calculate_accuracy_and_f1('./train_and_val_preds/meta-llama/Meta-Llama-3.1-8B-Instruct_fine_tuned_val_set.csv')
