"""
DateTime feature extraction with cyclical encoding.

sin_month = sin(2π · month / 12)
cos_month = cos(2π · month / 12)

Ensures that distance(Jan, Dec) ≈ distance(Jan, Feb).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DateTimeFeatureEngineer:
    """Extract temporal features from datetime columns."""

    def __init__(self, config) -> None:
        self.features: list[str] = getattr(config, "features", ["month", "day_of_week", "is_weekend"])
        self.cyclical: bool = getattr(config, "cyclical_encoding", True)
        self._input_cols: List[str] = []
        self._output_names: List[str] = []

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self._input_cols = list(X.columns)
        return self._extract(X, fit=True)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._extract(X[self._input_cols], fit=False)

    def _extract(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []
        names: List[str] = []

        for col in X.columns:
            dt = pd.to_datetime(X[col], errors="coerce")

            extractors = {
                "year":        lambda d: d.dt.year,
                "month":       lambda d: d.dt.month,
                "day":         lambda d: d.dt.day,
                "hour":        lambda d: d.dt.hour,
                "minute":      lambda d: d.dt.minute,
                "day_of_week": lambda d: d.dt.dayofweek,
                "quarter":     lambda d: d.dt.quarter,
            }

            for feat in self.features:
                if feat == "is_weekend":
                    vals = (dt.dt.dayofweek >= 5).astype(int).values.reshape(-1, 1)
                    parts.append(vals)
                    names.append(f"{col}_is_weekend")
                    continue

                ext_fn = extractors.get(feat)
                if ext_fn is None:
                    continue

                raw = ext_fn(dt).values.astype(float)

                if self.cyclical and feat in ("month", "day_of_week", "hour", "minute", "quarter"):
                    period = {
                        "month": 12, "day_of_week": 7,
                        "hour": 24, "minute": 60, "quarter": 4,
                    }[feat]
                    sin_v = np.sin(2 * np.pi * raw / period).reshape(-1, 1)
                    cos_v = np.cos(2 * np.pi * raw / period).reshape(-1, 1)
                    parts.extend([sin_v, cos_v])
                    names.extend([f"{col}_{feat}_sin", f"{col}_{feat}_cos"])
                else:
                    parts.append(raw.reshape(-1, 1))
                    names.append(f"{col}_{feat}")

        if fit:
            self._output_names = names
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)

    @property
    def feature_names(self) -> List[str]:
        return self._output_names
