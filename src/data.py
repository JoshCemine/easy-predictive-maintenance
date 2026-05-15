"""Data loading and splitting for the AI4I 2020 dataset."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HWF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def download_if_missing(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path}...")
        urllib.request.urlretrieve(DATASET_URL, path)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    download_if_missing(path)
    df = pd.read_csv(path)
    df = df.drop(columns=ID_COLUMNS + FAILURE_MODE_COLUMNS, errors="ignore")
    return df


def split(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
