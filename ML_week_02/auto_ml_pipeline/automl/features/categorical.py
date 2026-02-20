"""
Categorical feature encoding.

Encoding Strategy (by cardinality):
  2 (binary)      → Label (0/1)
  3-10 (low)      → One-Hot
  11-100 (medium) → Target Encoding (Bayesian smoothed)
  > 100 (high)    → Frequency Encoding
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CategoricalFeatureEngineer:
    """Automatic categorical encoding based on cardinality."""

    def __init__(self, config) -> None:
        self.high_card_threshold = getattr(config, "high_cardinality_threshold", 10)
        self._encoders: dict = {}
        self._col_order: List[str] = []
        self._output_names: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        X = X.copy()
        self._col_order = list(X.columns)
        parts: list[np.ndarray] = []
        names: List[str] = []

        for col in self._col_order:
            n_unique = X[col].nunique()
            filled = X[col].fillna("__missing__")

            if n_unique <= 2:
                enc = LabelEncoder()
                encoded = enc.fit_transform(filled.astype(str)).reshape(-1, 1)
                self._encoders[col] = ("label", enc)
                parts.append(encoded)
                names.append(col)

            elif n_unique <= self.high_card_threshold:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
                encoded = enc.fit_transform(filled.values.reshape(-1, 1))
                self._encoders[col] = ("ohe", enc)
                parts.append(encoded)
                ohe_names = [f"{col}_{cat}" for cat in enc.categories_[0]]
                names.extend(ohe_names)

            elif n_unique <= 100:
                if y is not None:
                    enc = BayesianTargetEncoder(smoothing=10.0)
                    encoded = enc.fit_transform(filled, y).reshape(-1, 1)
                    self._encoders[col] = ("target", enc)
                else:
                    enc = FrequencyEncoder()
                    encoded = enc.fit_transform(filled).reshape(-1, 1)
                    self._encoders[col] = ("frequency", enc)
                parts.append(encoded)
                names.append(f"{col}_enc")

            else:
                enc = FrequencyEncoder()
                encoded = enc.fit_transform(filled).reshape(-1, 1)
                self._encoders[col] = ("frequency", enc)
                parts.append(encoded)
                names.append(f"{col}_freq")

        self._output_names = names
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        for col in self._col_order:
            filled = X[col].fillna("__missing__")
            enc_type, enc = self._encoders[col]
            if enc_type == "label":
                try:
                    encoded = enc.transform(filled.astype(str)).reshape(-1, 1)
                except ValueError:
                    # Unseen labels → map to 0
                    mapping = {c: i for i, c in enumerate(enc.classes_)}
                    encoded = filled.astype(str).map(mapping).fillna(0).values.reshape(-1, 1)
            elif enc_type == "ohe":
                encoded = enc.transform(filled.values.reshape(-1, 1))
            elif enc_type == "target":
                encoded = enc.transform(filled).reshape(-1, 1)
            else:
                encoded = enc.transform(filled).reshape(-1, 1)
            parts.append(encoded)
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)

    @property
    def feature_names(self) -> List[str]:
        return self._output_names


# ── Encoder helpers ───────────────────────────────────────────────


class BayesianTargetEncoder:
    """
    Target encoding with Bayesian smoothing.

    encoded(c) = (count_c * mean_c + k * global_mean) / (count_c + k)
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing
        self._map: dict = {}
        self._global_mean: float = 0.0

    def fit_transform(self, X: pd.Series, y: pd.Series) -> np.ndarray:
        self._global_mean = float(y.mean())
        stats = pd.DataFrame({"cat": X, "y": y}).groupby("cat")["y"].agg(["mean", "count"])
        self._map = (
            (stats["count"] * stats["mean"] + self.smoothing * self._global_mean)
            / (stats["count"] + self.smoothing)
        ).to_dict()
        return X.map(self._map).fillna(self._global_mean).values

    def transform(self, X: pd.Series) -> np.ndarray:
        return X.map(self._map).fillna(self._global_mean).values


class FrequencyEncoder:
    """Encode categories by relative frequency in the training set."""

    def __init__(self) -> None:
        self._map: dict = {}

    def fit_transform(self, X: pd.Series) -> np.ndarray:
        self._map = (X.value_counts(normalize=True)).to_dict()
        return X.map(self._map).fillna(0.0).values

    def transform(self, X: pd.Series) -> np.ndarray:
        return X.map(self._map).fillna(0.0).values
