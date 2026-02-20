"""
Cross-validation helpers with optional stratification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """Configurable cross-validation runner."""

    def __init__(self, config) -> None:
        self.n_folds: int = getattr(config, "n_folds", 5)
        self.stratified: bool = getattr(config, "stratified", True)
        self.shuffle: bool = getattr(config, "shuffle", True)
        self.random_state: int = getattr(config, "random_state", 42)

    def get_splitter(self, task: str):
        if task == "classification" and self.stratified:
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        return KFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        task: str,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run k-fold CV and return mean / std / per-fold scores."""
        cv = self.get_splitter(task)
        if scoring is None:
            scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            "scoring": scoring,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "per_fold": scores.tolist(),
        }

    def predict_oof(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        task: str,
    ) -> np.ndarray:
        """Out-of-fold predictions."""
        cv = self.get_splitter(task)
        method = "predict_proba" if task == "classification" else "predict"
        try:
            return cross_val_predict(model, X, y, cv=cv, method=method, n_jobs=-1)
        except Exception:
            return cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=-1)
