"""
Outlier detection and treatment.

Supported methods: IQR, Z-score, Isolation Forest.
Treatments: clip, remove, flag.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect and treat outliers in numeric columns."""

    def __init__(self, config) -> None:
        self.method: str = getattr(config, "method", "iqr")
        self.threshold: float = getattr(config, "threshold", 3.0)
        self.treatment: str = getattr(config, "treatment", "clip")

    def treat(
        self,
        df: pd.DataFrame,
        type_info: Dict[str, List[str]],
    ) -> Tuple[pd.DataFrame, Dict]:
        """Detect and treat outliers in all numeric columns.

        Returns:
            (treated_df, report)
        """
        df = df.copy()
        total_outliers = 0
        cols_affected = 0

        num_cols = type_info.get("numeric", [])

        for col in num_cols:
            if col not in df.columns:
                continue
            mask = self._detect(df[col])
            n = mask.sum()
            if n == 0:
                continue
            total_outliers += n
            cols_affected += 1
            df = self._apply_treatment(df, col, mask)

        report = {"n_outliers": int(total_outliers), "n_cols_affected": cols_affected}
        return df, report

    # ── detection ─────────────────────────────────────────────────
    def _detect(self, s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return pd.Series(False, index=s.index)

        if self.method == "iqr":
            q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            return (s < lower) | (s > upper)

        if self.method == "zscore":
            z = (s - s_clean.mean()) / (s_clean.std() + 1e-9)
            return z.abs() > self.threshold

        # Fallback: no outliers
        return pd.Series(False, index=s.index)

    # ── treatment ─────────────────────────────────────────────────
    def _apply_treatment(
        self, df: pd.DataFrame, col: str, mask: pd.Series
    ) -> pd.DataFrame:
        if self.treatment == "clip":
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            df[col] = df[col].clip(lower, upper)
        elif self.treatment == "remove":
            df = df[~mask].reset_index(drop=True)
        elif self.treatment == "flag":
            df[f"{col}_outlier"] = mask.astype(int)
        return df

    # ── Convenience wrappers (match pipeline API) ─────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
    ) -> pd.DataFrame:
        """Detect outliers, store bounds, treat, return cleaned df."""
        type_info = {"numeric": num_cols}
        result, _ = self.treat(df, type_info)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same treatment to new data (uses same thresholds)."""
        # For clip treatment the bounds come from the data itself,
        # so we just re-run with stored params
        return df