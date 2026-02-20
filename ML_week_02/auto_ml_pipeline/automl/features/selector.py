"""
Automatic feature selection.

Pipeline:
  1. Variance threshold  (remove near-constant)
  2. Correlation filter   (remove highly correlated)
  3. Importance filter    (remove very-low-importance via quick RF)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Multi-stage feature selection."""

    def __init__(self, config) -> None:
        self.var_threshold: float = getattr(config, "variance_threshold", 0.01)
        self.corr_threshold: float = getattr(config, "correlation_threshold", 0.95)
        self.imp_threshold: float = getattr(config, "importance_threshold", 0.001)
        self._var_selector: Optional[VarianceThreshold] = None
        self._drop_corr: List[int] = []
        self._keep_mask: Optional[np.ndarray] = None
        self.selected_feature_names_: List[str] = []

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # 1. Variance threshold
        self._var_selector = VarianceThreshold(threshold=self.var_threshold)
        try:
            X = self._var_selector.fit_transform(X)
            mask = self._var_selector.get_support()
            names = [n for n, m in zip(names, mask) if m]
        except ValueError:
            pass  # all-zero variance edge case

        # Replace any remaining NaN/inf with 0 for downstream safety
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Correlation filter
        if X.shape[1] > 1:
            corr = np.corrcoef(X, rowvar=False)
            drop_idx: set[int] = set()
            for i in range(corr.shape[0]):
                if i in drop_idx:
                    continue
                for j in range(i + 1, corr.shape[1]):
                    if j in drop_idx:
                        continue
                    if abs(corr[i, j]) > self.corr_threshold:
                        drop_idx.add(j)
            keep = sorted(set(range(X.shape[1])) - drop_idx)
            X = X[:, keep]
            names = [names[i] for i in keep]
            self._drop_corr = list(drop_idx)

        # 3. Importance filter (quick RF, skip if too few rows/features)
        if y is not None and X.shape[1] > 1 and len(y) >= 30:
            try:
                is_clf = y.nunique() <= 20
                rf = (
                    RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
                    if is_clf
                    else RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
                )
                rf.fit(X, y)
                imps = rf.feature_importances_
                imp_mask = imps >= self.imp_threshold
                if imp_mask.sum() >= 1:
                    X = X[:, imp_mask]
                    names = [n for n, m in zip(names, imp_mask) if m]
            except Exception:
                pass  # graceful fallback

        self.selected_feature_names_ = names
        self._keep_mask = np.ones(X.shape[1], dtype=bool)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._var_selector is not None:
            X = self._var_selector.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self._drop_corr:
            keep = sorted(set(range(X.shape[1])) - set(self._drop_corr))
            X = X[:, keep]
        # Importance mask — clip to available columns
        if self._keep_mask is not None and len(self._keep_mask) <= X.shape[1]:
            X = X[:, : len(self._keep_mask)]
        return X
