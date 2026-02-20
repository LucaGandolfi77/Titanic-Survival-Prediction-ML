"""
Numeric feature engineering: scaling, polynomial expansion, binning.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)

_SCALER_MAP = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


class NumericFeatureEngineer:
    """Scale, optionally create polynomial / binned features."""

    def __init__(self, config) -> None:
        self.cfg = config
        self._scaler = None
        self._poly = None
        self._binner = None
        self._input_cols: List[str] = []
        self._output_names: List[str] = []

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        self._input_cols = list(X.columns)

        # Fill residual NaNs with 0 (safety net)
        X = X.fillna(0)
        arr = X.values.astype(np.float64)

        # Scaling
        scaler_cls = _SCALER_MAP.get(
            getattr(self.cfg.scaling, "method", "robust"), RobustScaler
        )
        self._scaler = scaler_cls()
        arr = self._scaler.fit_transform(arr)

        parts = [arr]
        names: List[str] = list(self._input_cols)

        # Polynomial features
        poly_cfg = self.cfg.polynomial
        if getattr(poly_cfg, "enabled", False):
            self._poly = PolynomialFeatures(
                degree=getattr(poly_cfg, "degree", 2),
                interaction_only=getattr(poly_cfg, "interaction_only", False),
                include_bias=False,
            )
            poly_arr = self._poly.fit_transform(arr)
            # Remove the originals (already in *arr*); keep only new ones
            n_orig = arr.shape[1]
            new_poly = poly_arr[:, n_orig:]

            max_feats = getattr(poly_cfg, "max_features", 50)
            if new_poly.shape[1] > max_feats:
                # Keep top-variance polynomial features
                var = new_poly.var(axis=0)
                top_idx = np.argsort(var)[-max_feats:]
                new_poly = new_poly[:, top_idx]

            parts.append(new_poly)
            names += [f"poly_{i}" for i in range(new_poly.shape[1])]

        # Binning
        bin_cfg = self.cfg.binning
        if getattr(bin_cfg, "enabled", False):
            n_bins = getattr(bin_cfg, "n_bins", 10)
            strategy = getattr(bin_cfg, "strategy", "quantile")
            self._binner = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None
            )
            binned = self._binner.fit_transform(arr)
            parts.append(binned)
            names += [f"{c}_bin" for c in self._input_cols]

        out = np.hstack(parts)
        self._output_names = names
        logger.debug(f"Numeric features: {arr.shape[1]} → {out.shape[1]}")
        return out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self._input_cols].fillna(0)
        arr = self._scaler.transform(X.values.astype(np.float64))
        parts = [arr]
        if self._poly is not None:
            poly_arr = self._poly.transform(arr)
            n_orig = arr.shape[1]
            new_poly = poly_arr[:, n_orig:]
            # Apply same column mask if we truncated
            parts.append(new_poly)
        if self._binner is not None:
            parts.append(self._binner.transform(arr))
        return np.hstack(parts)

    @property
    def feature_names(self) -> List[str]:
        return self._output_names
