"""
preprocessor.py – Data preprocessing operations (mirrors Weka filters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OrdinalEncoder,
)


class Preprocessor:
    """Stateless helper that applies transformations to DataFrames."""

    # ── Missing values ────────────────────────────────────────────────
    @staticmethod
    def remove_missing_rows(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        return df.dropna(subset=cols) if cols else df.dropna()

    @staticmethod
    def fill_missing(df: pd.DataFrame, col: str, strategy: str = "mean") -> pd.DataFrame:
        df = df.copy()
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "drop":
            df = df.dropna(subset=[col])
        else:
            df[col] = df[col].fillna(strategy)  # custom value
        return df

    # ── Encoding ──────────────────────────────────────────────────────
    @staticmethod
    def label_encode(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, LabelEncoder]:
        df = df.copy()
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return df, le

    @staticmethod
    def one_hot_encode(df: pd.DataFrame, col: str, max_cats: int = 20) -> pd.DataFrame:
        n_unique = df[col].nunique()
        if n_unique > max_cats:
            top = df[col].value_counts().head(max_cats).index
            df = df.copy()
            df[col] = df[col].where(df[col].isin(top), "Other")
        return pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)

    @staticmethod
    def ordinal_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df.copy()
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[col] = oe.fit_transform(df[[col]])
        return df

    # ── Scaling ───────────────────────────────────────────────────────
    @staticmethod
    def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df

    @staticmethod
    def normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df

    # ── Column operations ─────────────────────────────────────────────
    @staticmethod
    def remove_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df.drop(columns=cols, errors="ignore")

    @staticmethod
    def rename_column(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
        return df.rename(columns={old: new})

    @staticmethod
    def cast_column(df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
        df = df.copy()
        if dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "string":
            df[col] = df[col].astype(str)
        elif dtype == "category":
            df[col] = df[col].astype("category")
        return df

    # ── Discretize (bin numeric into categories) ──────────────────────
    @staticmethod
    def discretize(df: pd.DataFrame, col: str, bins: int = 5,
                   labels: list[str] | None = None) -> pd.DataFrame:
        df = df.copy()
        df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels)
        return df

    # ── Outlier removal ───────────────────────────────────────────────
    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        return df[(df[col] >= lower) & (df[col] <= upper)]

    # ── Sample ────────────────────────────────────────────────────────
    @staticmethod
    def sample(df: pd.DataFrame, n: int | None = None,
               frac: float | None = None, seed: int = 42) -> pd.DataFrame:
        if n is not None:
            return df.sample(n=min(n, len(df)), random_state=seed)
        if frac is not None:
            return df.sample(frac=frac, random_state=seed)
        return df
