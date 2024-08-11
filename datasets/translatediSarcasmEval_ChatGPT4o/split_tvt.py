import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dataset = pd.read_csv('translated_dataset.csv')
target_col = 'label'

for _ in range(10):
    dataset = shuffle(dataset, random_state=42)

train_set, test_set = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], shuffle=True, random_state=42)

train_set, val_set = train_test_split(train_set, test_size=0.25, stratify=train_set['label'], shuffle=True, random_state=42)


test_set.to_csv('./train_val_test_split/test_set.csv', index=False)
val_set.to_csv('./train_val_test_split/val_set.csv', index=False)
train_set.to_csv('./train_val_test_split/train_set.csv', index=False)

print(test_set['label'].value_counts())
print(val_set['label'].value_counts())
print(train_set['label'].value_counts())
