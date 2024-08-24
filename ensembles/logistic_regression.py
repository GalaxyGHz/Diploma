from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

def preprocess_dataset(df):

    columns_to_check = [x for x in df if not (x == 'label' or x == 'custom_id')]
    print(columns_to_check)

    df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=columns_to_check)

    return df

def train_logistic_regression():
    train_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/train_set_predictions.csv'))
    val_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/validation_set_predictions.csv'))
    test_dataset = preprocess_dataset(pd.read_csv('./combined_model_predictions/test_set_predictions.csv'))

    y_train = train_dataset['label']  
    X_train = train_dataset.drop(columns=['label', 'custom_id'], axis=1) 

    y_val = val_dataset['label']  
    X_val = val_dataset.drop(columns=['label', 'custom_id'], axis=1) 

    y_test = test_dataset['label']  
    X_test = test_dataset.drop(columns=['label', 'custom_id'], axis=1) 

    best_model = None
    best_accuracy = 0
    C_range = np.logspace(-5, 1, 100)

    accuracies = []
    for C in tqdm(C_range):
        model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            

    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    print('accuracy:', accuracy, 'f1:', f1)

    # Accuracy
    max_accuracy_index = np.argmax(accuracies)
    print("Best C:", C_range[max_accuracy_index])
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.log10(C_range), y=accuracies, marker='o', color=(0.5, 0.68, 0.99))
    plt.scatter(np.log10(C_range[max_accuracy_index]), accuracies[max_accuracy_index], color=(0.9, 0.5, 0.6), s=100, zorder=5)
    plt.text(-3.2, 0.77, f'Max Accuracy: {np.round(max(accuracies), 4)}')
    plt.xlabel('Log10(C)')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Log of Inverse Regularization Strength (C)')
    plt.grid(True)
    plt.show()
    plt.savefig(f"./figures/accuracy_based_on_C.pdf")


    # Weights
    print(best_model.coef_[0])
    weights_df = pd.DataFrame({
        'Model': [x.replace('_probability', "").replace('_prediction', "") for x in best_model.feature_names_in_],
        'Weight': best_model.coef_[0]
    })

    # weights_df['Model'] = weights_df['Model'].apply(lambda x: x[:-12])
    weights_df = weights_df.replace('EMBEDDIA', "SloBERTa")

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Weight', y='Model', data=weights_df, color=(0.5, 0.68, 0.99))

    # Customize the plot
    plt.xlabel('Weight')
    plt.ylabel('Model')
    plt.title('L2 Logistic Regression Weights')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(f"./figures/log_reg_weigths.pdf")


if __name__ == '__main__':

    train_logistic_regression()