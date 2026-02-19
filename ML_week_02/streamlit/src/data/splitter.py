"""
splitter.py – Train / test split strategies.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_STRATEGIES = ["holdout", "stratified"]


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    strategy: str = "holdout",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into X_train, X_test, y_train, y_test.

    Parameters
    ----------
    strategy : str
        "holdout" – plain random split
        "stratified" – stratified on target (classification only)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if strategy == "stratified" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
