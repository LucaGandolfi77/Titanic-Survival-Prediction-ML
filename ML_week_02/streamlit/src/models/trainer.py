"""
trainer.py – Training orchestration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .registry import create_estimator, get_model


@dataclass
class TrainResult:
    """Container for a single model's training outcome."""
    model_key: str
    display_name: str
    estimator: BaseEstimator
    params: Dict[str, Any]
    train_time_sec: float
    task: str


def train_single(
    model_key: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    params: Dict[str, Any] | None = None,
) -> TrainResult:
    """Train a single model and return its TrainResult."""
    info = get_model(model_key)
    estimator = create_estimator(model_key, params)

    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    return TrainResult(
        model_key=model_key,
        display_name=info.display_name,
        estimator=estimator,
        params=estimator.get_params(),
        train_time_sec=round(elapsed, 4),
        task=info.task,
    )


def train_multiple(
    model_keys: List[str],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    params_map: Dict[str, Dict[str, Any]] | None = None,
    progress_callback=None,
) -> List[TrainResult]:
    """Train several models sequentially.

    *progress_callback(i, total, model_name)* is called after each model
    finishes — handy for Streamlit progress bars.
    """
    params_map = params_map or {}
    results: List[TrainResult] = []

    for i, key in enumerate(model_keys):
        result = train_single(key, X_train, y_train, params_map.get(key))
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(model_keys), result.display_name)

    return results
