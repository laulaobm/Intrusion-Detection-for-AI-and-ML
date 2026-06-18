import pandas as pd

def load_preprocessed_data(task_type='binary'):
    X_test = pd.read_parquet('../../data/X_test.parquet')

    if task_type == 'binary':
        X_train = pd.read_parquet('../../data/X_train_binary_resampled.parquet')
        y_train = pd.read_parquet('../../data/y_train_binary_resampled.parquet')['target']
        y_test = pd.read_parquet('../../data/y_test_binary.parquet')['target']
    elif task_type == 'multi':
        X_train = pd.read_parquet('../../data/X_train_multi_resampled.parquet')
        y_train = pd.read_parquet('../../data/y_train_multi_resampled.parquet')['target']
        y_test = pd.read_parquet('../../data/y_test_multi.parquet')['target']
    else:
        raise ValueError("task_type must be 'binary' or 'multi'")

    return X_train, X_test, y_train, y_test