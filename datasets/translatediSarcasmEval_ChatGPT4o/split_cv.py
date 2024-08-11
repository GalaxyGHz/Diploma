import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

dataset = pd.read_csv('translated_dataset.csv')
target_col = 'label'

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for _ in range(10):
    dataset = shuffle(dataset, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset[target_col])):
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]
    
    train_file = f'cv_split/train_fold_{fold}.csv'
    test_file = f'cv_split/test_fold_{fold}.csv'
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f'Saved {train_file}/{len(train_df)} and {test_file}/{len(test_df)}')