import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def calculate_accuracy_and_f1(path_to_preds='.'):
    predictions = pd.read_csv(path_to_preds)

    label = predictions['label']
    preds = predictions['prediction']

    f1 = f1_score(y_true=label, y_pred=preds)
    acccuracy = accuracy_score(y_true=label, y_pred=preds)
    print(path_to_preds, ":", "accuracy:", acccuracy, "f1:", f1)



if __name__ == '__main__':
    calculate_accuracy_and_f1('./results/gpt3_no_fine_tuning.csv')
    calculate_accuracy_and_f1('./results/gpt4_no_fine_tuning.csv')
    calculate_accuracy_and_f1('./results/gpt3_fine_tuning.csv')