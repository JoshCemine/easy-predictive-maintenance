"""Smoke tests for data loading + splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, TARGET, load_dataframe, split

DATA_PATH = Path("data/ai4i2020.csv")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataframe(DATA_PATH)


def test_loaded_dataframe_has_expected_columns(df: pd.DataFrame) -> None:
    for col in FEATURE_COLUMNS + [TARGET]:
        assert col in df.columns, f"missing column: {col}"


def test_no_nans_in_features_or_target(df: pd.DataFrame) -> None:
    assert df[FEATURE_COLUMNS + [TARGET]].isna().sum().sum() == 0


def test_target_is_binary(df: pd.DataFrame) -> None:
    assert set(df[TARGET].unique()) == {0, 1}


def test_split_is_stratified_and_disjoint(df: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split(df, test_size=0.2, random_state=42)
    # Both classes present in both splits
    assert y_train.nunique() == 2
    assert y_test.nunique() == 2
    # Stratification preserves positive rate (within tolerance)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    # No index overlap
    overlap = set(X_train.index) & set(X_test.index)
    assert not overlap, f"train/test index overlap: {len(overlap)} rows"


def test_id_columns_were_dropped(df: pd.DataFrame) -> None:
    assert "UDI" not in df.columns
    assert "Product ID" not in df.columns
