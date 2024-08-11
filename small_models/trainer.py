import numpy as np
import pandas as pd
import json
import torch
import random
import optuna
import torch.optim as optim

import evaluate
from transformers import set_seed
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

set_seed(42)
random.seed(10)
np.random.seed(42)
torch.manual_seed(42)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32, 16, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 0.1, 0.5, log=True),
        "optim": trial.suggest_categorical("optim",["adamw_torch"])
    }

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions.argmax(-1)

    accuracy = evaluate.load("accuracy")
    f1_score = evaluate.load("f1")
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "f1": f1_score.compute(predictions=predictions, references=labels),
    }

@torch.no_grad
def get_test_results(model_name, tokenizer, model):
    test_set = pd.read_csv('../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/test_set.csv')

    test_set['probability'] = test_set['translation'].apply(
        lambda value: 
        torch.nn.functional.softmax(model(**tokenizer(value, padding=True, truncation=True, max_length=256, return_tensors='pt')).logits, dim=-1)[:, 1].item()
        )
    test_set['prediction'] = test_set['probability'].apply(lambda value: np.where(value >= 0.5, 1, 0))

    test_set = test_set.drop(columns=['text', 'translation'])
    test_set.to_csv(f"./results/{model_name.split('/')[0]}_predictions.csv", encoding='utf-8', index=False, header=True)
    return


class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

def prepare_data(tokenizer, path_to_data='.'):
    dataset = pd.read_csv(path_to_data)
    X = dataset['translation'].to_list()
    y = dataset['label'].to_list()

    tokenizer_X = tokenizer(X, padding=True, truncation=True, max_length=256)
    dataset = SarcasmDataset(tokenizer_X, y)
    return dataset


def run_training(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/train_set.csv'
    validation_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/val_set.csv'
    testing_path = f'../datasets/translatediSarcasmEval_ChatGPT4o/train_val_test_split/test_set.csv'

    train_set = prepare_data(tokenizer, training_path)
    val_set = prepare_data(tokenizer, validation_path)
    test_set = prepare_data(tokenizer, testing_path)

    STEPS = 50
    
    training_args = TrainingArguments(
        save_strategy="steps",
        save_steps=STEPS,
        save_total_limit=1,
        output_dir=f"./best_checkpoints/{model_name}/",
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=STEPS,
        num_train_epochs=5,
        log_level="info",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        data_seed=42,
    )


    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    trainer = Trainer(
        model=None,
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
        sampler=optuna.samplers.TPESampler(seed=42),
        compute_objective=lambda metrics: metrics["eval_loss"],
    )

    for n, v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()

    trainer.save_model(f"./best_checkpoints/{model_name}/best_model")
    with open(f"./best_checkpoints/{model_name}/best_model/hyperparameters.json", "w") as hp_file:
        json.dump(best_trial.hyperparameters, hp_file)

    tokenizer = AutoTokenizer.from_pretrained(f"./best_checkpoints/{model_name}/best_model")
    model = AutoModelForSequenceClassification.from_pretrained(f"./best_checkpoints/{model_name}/best_model")
    get_test_results(model_name, tokenizer, model)

if __name__ == "__main__":
    MODELS = [
        "bert-base-multilingual-cased",
        "EMBEDDIA/sloberta",
        "xlm-roberta-base",
        "xlm-roberta-large",
    ]

    for model in MODELS:
        run_training(model)