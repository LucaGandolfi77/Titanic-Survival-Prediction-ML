"""
data_loader.py – Load datasets from various sources for the XAI dashboard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@st.cache_data(show_spinner="Loading dataset …")
def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a CSV file with caching."""
    return pd.read_csv(path)


def load_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Read a Streamlit UploadedFile (CSV or Excel)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


def list_available_datasets() -> List[str]:
    """List CSV files in the data directory."""
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.csv"))


def get_dataset_path(name: str) -> Path:
    """Return full path for a dataset name."""
    return DATA_DIR / name


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick overview stats for a dataframe."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": len(df.select_dtypes(include=["object", "category"]).columns),
        "missing_pct": f"{df.isnull().mean().mean():.1%}",
        "memory_mb": f"{df.memory_usage(deep=True).sum() / 1e6:.2f}",
    }


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X, y.  Encode categoricals automatically."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Simple encoding for categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    if y.dtype == "object" or y.dtype.name == "category":
        y = y.astype("category").cat.codes

    # Fill missing
    X = X.fillna(X.median(numeric_only=True))

    return X, y
