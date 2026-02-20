"""
Smart missing-value imputation with strategy selection by missing %.

Strategy Matrix
───────────────
  Missing %   │ Numeric        │ Categorical
  ───────────────────────────────────────────
  < 5 %       │ median         │ most_frequent
  5 – 20 %    │ KNN (5-nn)     │ most_frequent
  20 – 70 %   │ constant -999  │ constant "missing"
  > 70 %      │ DROP column    │ DROP column
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MissingValueHandler:
    """Select and apply imputation strategies per column."""

    def __init__(self, config) -> None:
        self.drop_col_threshold: float = getattr(config, "drop_column_threshold", 0.7)
        self.drop_row_threshold: float = getattr(config, "drop_row_threshold", 0.5)
        self._fitted: dict[str, Any] = {}

    def handle(
        self,
        df: pd.DataFrame,
        type_info: Dict[str, List[str]],
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Impute or drop missing values.

        Returns:
            (cleaned_df, report_dict)
        """
        df = df.copy()
        report: Dict[str, Any] = {
            "n_dropped_cols": 0,
            "n_imputed_cols": 0,
            "strategies": {},
        }

        numeric_set = set(type_info.get("numeric", []))

        for col in list(df.columns):
            pct = df[col].isna().mean()
            if pct == 0:
                continue

            # Drop high-missing columns
            if pct > self.drop_col_threshold:
                df.drop(columns=[col], inplace=True)
                report["n_dropped_cols"] += 1
                report["strategies"][col] = f"DROPPED ({pct:.0%})"
                # Also remove from type_info to keep everything in sync
                for bucket in type_info.values():
                    if col in bucket:
                        bucket.remove(col)
                continue

            col_type = "numeric" if col in numeric_set else "categorical"
            strategy = self._pick_strategy(pct, col_type)
            report["strategies"][col] = strategy
            report["n_imputed_cols"] += 1

            if fit:
                imp = self._make_imputer(strategy, col_type)
                df[col] = imp.fit_transform(df[[col]]).ravel()
                self._fitted[col] = imp
            else:
                if col in self._fitted:
                    df[col] = self._fitted[col].transform(df[[col]]).ravel()

        return df, report

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _pick_strategy(pct: float, col_type: str) -> str:
        if pct < 0.05:
            return "median" if col_type == "numeric" else "most_frequent"
        if pct < 0.20:
            return "knn" if col_type == "numeric" else "most_frequent"
        return "constant"

    @staticmethod
    def _make_imputer(strategy: str, col_type: str):
        if strategy == "knn":
            return KNNImputer(n_neighbors=5)
        if strategy == "constant":
            fv = -999 if col_type == "numeric" else "missing"
            return SimpleImputer(strategy="constant", fill_value=fv)
        return SimpleImputer(strategy=strategy)

    # ── Convenience wrappers (match pipeline API) ─────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        type_info: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Fit imputers and transform. Returns cleaned DataFrame."""
        result, _ = self.handle(df, type_info, fit=True)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers to new data."""
        df = df.copy()
        for col, imp in self._fitted.items():
            if col in df.columns and df[col].isna().any():
                df[col] = imp.transform(df[[col]]).ravel()
        return df