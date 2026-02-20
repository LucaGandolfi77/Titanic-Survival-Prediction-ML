"""
Statistical profiling of a DataFrame.

Produces a summary dict used by the HTML report generator.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProfiler:
    """Compute descriptive statistics and distribution info."""

    def __init__(self, config=None) -> None:
        self.config = config

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a profiling dictionary.

        Keys:
            shape, dtypes, missing_pct, numeric_stats, correlations,
            categorical_top, memory_mb
        """
        prof: Dict[str, Any] = {}

        prof["shape"] = {"rows": len(df), "cols": len(df.columns)}
        prof["dtypes"] = df.dtypes.astype(str).to_dict()

        # Missing
        missing = df.isna().mean()
        prof["missing_pct"] = missing[missing > 0].to_dict()

        # Numeric stats
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            prof["numeric_stats"] = num_df.describe().to_dict()
        else:
            prof["numeric_stats"] = {}

        # Correlations (top-k pairs)
        if len(num_df.columns) >= 2:
            corr = num_df.corr()
            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            pairs.columns = ["feat_a", "feat_b", "corr"]
            pairs["abs_corr"] = pairs["corr"].abs()
            prof["top_correlations"] = (
                pairs.nlargest(10, "abs_corr")[["feat_a", "feat_b", "corr"]]
                .to_dict(orient="records")
            )
        else:
            prof["top_correlations"] = []

        # Categorical top values
        cat_df = df.select_dtypes(include="object")
        prof["categorical_top"] = {}
        for c in cat_df.columns:
            prof["categorical_top"][c] = (
                cat_df[c].value_counts().head(5).to_dict()
            )

        prof["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

        logger.info(
            f"Profile: {prof['shape']['rows']} rows × {prof['shape']['cols']} cols, "
            f"{prof['memory_mb']} MB"
        )
        return prof
