"""
statistics.py – Descriptive statistics for the EDA page.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() that includes dtype, missing count & unique count."""
    stats = df.describe(include="all").T
    stats["dtype"] = df.dtypes
    stats["missing"] = df.isnull().sum()
    stats["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    stats["unique"] = df.nunique()
    stats["memory_bytes"] = df.memory_usage(deep=True).drop("Index").values
    return stats


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats for numeric columns only."""
    num = df.select_dtypes("number")
    if num.empty:
        return pd.DataFrame()
    summary = num.describe().T
    summary["skew"] = num.skew()
    summary["kurtosis"] = num.kurtosis()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary.round(4)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats for categorical columns."""
    cat = df.select_dtypes(["object", "category"])
    if cat.empty:
        return pd.DataFrame()
    records = []
    for col in cat.columns:
        vc = cat[col].value_counts(dropna=False)
        records.append({
            "column": col,
            "unique": cat[col].nunique(),
            "top_value": vc.index[0] if len(vc) else None,
            "top_freq": vc.iloc[0] if len(vc) else 0,
            "missing": cat[col].isnull().sum(),
        })
    return pd.DataFrame(records).set_index("column")


def target_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Value-counts table for the target column."""
    vc = df[target_col].value_counts(dropna=False).reset_index()
    vc.columns = ["value", "count"]
    vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(2)
    return vc
