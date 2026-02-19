"""
predictor.py – Unified prediction wrapper.

Wraps any sklearn-compatible estimator to provide a consistent prediction
interface, including class probabilities when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class PredictionResult:
    """Container for a single prediction (or batch)."""
    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None


class Predictor:
    """Thin wrapper around an estimator for consistent interface."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self._is_classifier = hasattr(estimator, "predict_proba")
        self.classes_ = getattr(estimator, "classes_", None)

    # ── Batch ─────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> PredictionResult:
        labels = self.estimator.predict(X)
        proba = None
        if self._is_classifier:
            try:
                proba = self.estimator.predict_proba(X)
            except Exception:
                pass
        return PredictionResult(labels=labels, probabilities=proba, classes=self.classes_)

    # ── Single row ────────────────────────────────────────────

    def predict_single(self, row: dict | pd.Series) -> PredictionResult:
        if isinstance(row, dict):
            row = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            row = row.to_frame().T
        return self.predict(row)

    # ── Helpers ───────────────────────────────────────────────

    @property
    def is_classifier(self) -> bool:
        return self._is_classifier

    def feature_names(self) -> list[str]:
        """Try to extract feature names from the estimator."""
        if hasattr(self.estimator, "feature_names_in_"):
            return list(self.estimator.feature_names_in_)
        if hasattr(self.estimator, "get_booster"):  # XGBoost
            booster = self.estimator.get_booster()
            return booster.feature_names or []
        return []
