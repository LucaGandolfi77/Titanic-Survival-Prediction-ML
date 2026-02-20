"""
Automatic column type detection.

Column Type Taxonomy
────────────────────
- numeric   : int/float with continuous distribution
- categorical: low-cardinality string/int columns
- datetime  : date/time strings or datetime dtype
- text      : high-cardinality strings with multiple words
- boolean   : binary (0/1, True/False, yes/no)
- id        : high-cardinality unique identifiers (drop)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TypeDetector:
    """
    Detect column types for routing to the correct feature engineer.

    Decision tree per column:
      1. Already datetime dtype? → datetime
      2. Parseable as datetime string? → datetime
      3. Numeric dtype?
         • unique-ratio ≤ threshold & nunique ≤ 20 → categorical
         • nunique == nrows (int) → id
         • else → numeric
      4. Object dtype?
         • all values boolean-like → boolean
         • average word count > threshold → text
         • unique-ratio > 0.9 → id
         • else → categorical
    """

    def __init__(self, config=None) -> None:
        self.numeric_threshold: float = getattr(config, "numeric_threshold", 0.05) if config else 0.05
        self.text_threshold: int = getattr(config, "text_threshold", 50) if config else 50
        self.datetime_formats: list[str] = (
            getattr(config, "datetime_formats", ["%Y-%m-%d", "%d/%m/%Y"]) if config
            else ["%Y-%m-%d", "%d/%m/%Y"]
        )

    # ── public API ────────────────────────────────────────────────
    def detect(
        self, df: pd.DataFrame, target: str | None = None
    ) -> Dict[str, List[str]]:
        """Classify every column (except *target*) into a type bucket."""
        types: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "boolean": [],
            "id": [],
            "drop": [target] if target else [],
        }

        for col in df.columns:
            if col == target:
                continue
            ctype = self._detect_single(df[col], len(df))
            types[ctype].append(col)

        logger.info(
            "Type detection: "
            + ", ".join(f"{k}={len(v)}" for k, v in types.items() if v and k != "drop")
        )
        return types

    # ── internals ─────────────────────────────────────────────────
    def _detect_single(self, s: pd.Series, n_rows: int) -> str:
        non_null = s.dropna()
        if len(non_null) == 0:
            return "drop"

        n_unique = non_null.nunique()
        dtype = s.dtype

        # Boolean
        if self._is_boolean(non_null, dtype):
            return "boolean"

        # Datetime
        if self._is_datetime(s):
            return "datetime"

        # Numeric
        if pd.api.types.is_numeric_dtype(dtype):
            ratio = n_unique / max(n_rows, 1)
            if ratio <= self.numeric_threshold and n_unique <= 20:
                return "categorical"
            if n_unique == n_rows and pd.api.types.is_integer_dtype(dtype):
                return "id"
            return "numeric"

        # Object / string
        if dtype == "object":
            # Try numeric coerce
            coerced = pd.to_numeric(non_null, errors="coerce")
            if coerced.notna().mean() > 0.9:
                return "numeric"
            avg_words = non_null.astype(str).str.split().str.len().mean()
            if avg_words > self.text_threshold:
                return "text"
            ratio = n_unique / max(n_rows, 1)
            if ratio > 0.9:
                return "id"
            return "categorical"

        return "numeric"  # fallback

    @staticmethod
    def _is_boolean(non_null: pd.Series, dtype) -> bool:
        if dtype == bool:
            return True
        if dtype == "object":
            vals = set(non_null.astype(str).str.lower().unique())
            return vals.issubset({"true", "false", "0", "1", "yes", "no"})
        if pd.api.types.is_integer_dtype(dtype) and set(non_null.unique()).issubset({0, 1}):
            return True
        return False

    def _is_datetime(self, s: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if s.dtype != "object":
            return False
        sample = s.dropna().head(10)
        for fmt in self.datetime_formats:
            try:
                pd.to_datetime(sample, format=fmt)
                return True
            except (ValueError, TypeError):
                continue
        try:
            pd.to_datetime(sample)
            return True
        except (ValueError, TypeError):
            return False
