import pandas as pd
import transformers
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

MODELS = [
    './best_checkpoints/bert-base-multilingual-cased/best_model/',
    './best_checkpoints/EMBEDDIA/sloberta/best_model/',
    './best_checkpoints/xlm-roberta-base/best_model/',
    './best_checkpoints/xlm-roberta-large/best_model/',
]  

training_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/train_set.csv'
validation_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/val_set.csv'
testing_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/test_set.csv'

train_set = pd.read_csv(training_path)
val_set = pd.read_csv(validation_path)

for model_path in MODELS:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model_name = model_path.split('/')[2]

    train_set[model_name + '_probability'] = train_set['translation'].apply(
        lambda value: 
        torch.nn.functional.softmax(model(**tokenizer(value, padding=True, truncation=True, max_length=256, return_tensors='pt')).logits, dim=-1)[:, 1].item()
        )
    train_set[model_name + '_prediction'] = train_set[model_name + '_probability'].apply(lambda value: np.where(value >= 0.5, 1, 0))

    val_set[model_name + '_probability'] = val_set['translation'].apply(
        lambda value: 
        torch.nn.functional.softmax(model(**tokenizer(value, padding=True, truncation=True, max_length=256, return_tensors='pt')).logits, dim=-1)[:, 1].item()
        )
    val_set[model_name + '_prediction'] = val_set[model_name + '_probability'].apply(lambda value: np.where(value >= 0.5, 1, 0))

train_set = train_set.drop(columns=['text', 'translation'])
train_set.to_csv(f"./results/train_set_predictions.csv", encoding='utf-8', index=False, header=True)

val_set = val_set.drop(columns=['text', 'translation'])
val_set.to_csv(f"./results/validation_set_predictions.csv", encoding='utf-8', index=False, header=True)