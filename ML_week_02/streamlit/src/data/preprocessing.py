"""
preprocessing.py – Missing-value handling, encoding & scaling pipelines.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# ── Missing-value strategies ─────────────────────────────────

IMPUTE_STRATEGIES_NUM = ["mean", "median", "most_frequent", "constant"]
IMPUTE_STRATEGIES_CAT = ["most_frequent", "constant"]


def impute_missing(
    df: pd.DataFrame,
    num_strategy: str = "median",
    cat_strategy: str = "most_frequent",
    fill_value: Optional[str] = None,
) -> pd.DataFrame:
    """Impute missing values in-place-style (returns new frame)."""
    df = df.copy()
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(["object", "category"]).columns

    if len(num_cols):
        imp = SimpleImputer(strategy=num_strategy, fill_value=0 if num_strategy == "constant" else None)
        df[num_cols] = imp.fit_transform(df[num_cols])

    if len(cat_cols):
        imp = SimpleImputer(
            strategy=cat_strategy,
            fill_value=fill_value or "missing",
        )
        df[cat_cols] = imp.fit_transform(df[cat_cols])

    return df


# ── Encoding strategies ──────────────────────────────────────

ENCODING_STRATEGIES = ["one_hot", "label", "ordinal"]


def encode_categoricals(
    df: pd.DataFrame,
    strategy: str = "one_hot",
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Encode categorical features.  *target_col* is excluded from encoding."""
    df = df.copy()
    cat_cols = [
        c for c in df.select_dtypes(["object", "category"]).columns
        if c != target_col
    ]
    if not cat_cols:
        return df

    if strategy == "one_hot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    elif strategy == "label":
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
    elif strategy == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    else:
        raise ValueError(f"Unknown encoding strategy: {strategy}")

    return df


# ── Scaling strategies ────────────────────────────────────────

SCALING_STRATEGIES = ["standard", "minmax", "none"]


def scale_features(
    df: pd.DataFrame,
    strategy: str = "standard",
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Scale numeric features. *target_col* is excluded."""
    if strategy == "none":
        return df

    df = df.copy()
    num_cols = [
        c for c in df.select_dtypes("number").columns
        if c != target_col
    ]
    if not num_cols:
        return df

    scaler = StandardScaler() if strategy == "standard" else MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ── Full preprocessing pipeline (convenience) ────────────────

def preprocess(
    df: pd.DataFrame,
    target_col: str,
    num_impute: str = "median",
    cat_impute: str = "most_frequent",
    encoding: str = "one_hot",
    scaling: str = "standard",
) -> pd.DataFrame:
    """Run the complete preprocessing chain."""
    df = impute_missing(df, num_strategy=num_impute, cat_strategy=cat_impute)
    df = encode_categoricals(df, strategy=encoding, target_col=target_col)
    df = scale_features(df, strategy=scaling, target_col=target_col)
    return df
