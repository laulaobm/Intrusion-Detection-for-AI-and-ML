import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

def load_preprocessed_data(task_type='binary'):
    X_test = pd.read_parquet(DATA_DIR / 'X_test.parquet')

    if task_type == 'binary':
        X_train = pd.read_parquet(DATA_DIR / 'X_train_binary_resampled.parquet')
        y_train = pd.read_parquet(DATA_DIR / 'y_train_binary_resampled.parquet')['target']
        y_test = pd.read_parquet(DATA_DIR / 'y_test_binary.parquet')['target']

    elif task_type == 'multi':
        X_train = pd.read_parquet(DATA_DIR / 'X_train_multi_resampled.parquet')
        y_train = pd.read_parquet(DATA_DIR / 'y_train_multi_resampled.parquet')['target']
        y_test = pd.read_parquet(DATA_DIR / 'y_test_multi.parquet')['target']

    else:
        raise ValueError("task_type must be 'binary' or 'multi'")

    return X_train, X_test, y_train, y_test