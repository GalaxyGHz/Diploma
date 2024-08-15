from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

def preprocess_dataset(df):
    columns_to_check = [x for x in df if not (x == 'label' or x == 'custom_id')]

    df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=columns_to_check)

    return df

def hard_voting(X, y_true):
    X = X.apply(lambda x: np.where(x >= 0.5, 1.0, 0.0))

    n_models = X.shape[1] 
    X['vote_sum'] = X.sum(axis=1)
    X['majority_vote'] = (X['vote_sum'] > (n_models / 2)).astype(int)

    y_pred = X['majority_vote']
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('Hard voting: accuracy:', accuracy, 'f1:', f1)

def soft_voting(X, y_true):
    X['overall_probability'] = X.mean(axis=1)
    X['overall_prediction'] = np.where(X['overall_probability'] > 0.5, 1, 0)

    y_pred = X['overall_prediction']
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('Soft voting: accuracy:', accuracy, 'f1:', f1)


trial = 1
def mixed_voting(input_X, y_true):
    global trial
    X_original = input_X.copy()
    X = input_X.copy()
    X_predictions = X_original.apply(lambda x: np.where(x >= 0.5, 1.0, 0.0))
    n_models = X_original.shape[1]
    X['vote_sum'] = X_predictions.sum(axis=1)
    X['majority_vote'] = (X['vote_sum'] > (n_models / 2)).astype(int)

    X['soft_vote'] = np.where(X_original.mean(axis=1) > 0.5, 1, 0)

    accuracies = []
    for n in range(12):
        X['prediction'] = X.apply(lambda row: 
            row['majority_vote'] if abs(row['vote_sum'] - (n_models - row['vote_sum'])) > n else row['soft_vote'],
            axis=1)

        y_pred = X['prediction']
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        accuracies.append(accuracy)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        print(f'Mixed voting {n}: accuracy:', accuracy, 'f1:', f1)


    data_df = pd.DataFrame({
        'n': [i for i in range(12)],
        'accuracy': accuracies,
    })

    hard_df = pd.DataFrame({
        'n': [0],
        'accuracy': [accuracies[0]],
    })

    soft_df = pd.DataFrame({
        'n': [11],
        'accuracy': [accuracies[-1]],
    })

    # Accuracy
    marker_size = 100
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=data_df, x='n', y='accuracy', marker='o', color=(0.5, 0.68, 0.99), label='mixed voting', zorder=1)
    sns.scatterplot(data=hard_df, x='n', y='accuracy', color=(0.9, 0.5, 0.6), s=marker_size, label="hard voting", marker='o', ax=ax, zorder=2)
    sns.scatterplot(data=soft_df, x='n', y='accuracy', color=(1.0, 0.0, 0.0), label="soft voting", marker='o', ax=ax, zorder=2)
    if trial == 1:
        plt.text(0.14, 0.66965, f'Max Accuracy: {np.round(max(accuracies), 4)}')
    else:
        plt.text(0.14, 0.687, f'Max Accuracy: {np.round(max(accuracies), 4)}')
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Cut-off Point')
    plt.grid(True)
    plt.show()
    plt.savefig(f"./figures/accuracy_based_on_cut-off-trial_{trial}.pdf")
    trial += 1

def voting_accuracy():
    train_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/train_set_predictions.csv'))
    val_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/validation_set_predictions.csv'))
    test_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/test_set_predictions.csv'))


    y_train = train_dataset['label']  
    X_train = train_dataset.drop(columns=['label', 'custom_id'], axis=1) 

    y_val = val_dataset['label']  
    X_val = val_dataset.drop(columns=['label', 'custom_id'], axis=1) 

    y_test = test_dataset['label']  
    X_test = test_dataset.drop(columns=['label', 'custom_id'], axis=1)


    hard_voting(X_test.drop(columns=['BERT-BASE-MULTILINGUAL-CASED_probability']), y_test)
    soft_voting(X_test.drop(columns=['BERT-BASE-MULTILINGUAL-CASED_probability']), y_test)
    mixed_voting(X_test.drop(columns=['BERT-BASE-MULTILINGUAL-CASED_probability']), y_test)
    mixed_voting(X_test[[
        'GPT-3.5-TURBO-0125_fine_tuned_probability',
        'GPT-4o-2024-05-13_probability',
        'META-Llama-3.1-405B-INSTRUCT_prediction',
        'META-Llama-3.1-70B-INSTRUCT_prediction',
        'META-Llama-3.1-70B-INSTRUCT_fine_tuned_prediction',
    ]], y_test)
    



if __name__ == '__main__':

    voting_accuracy()