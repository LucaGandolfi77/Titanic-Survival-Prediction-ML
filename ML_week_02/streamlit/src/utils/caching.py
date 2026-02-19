"""
caching.py – Streamlit caching strategies.

Thin wrappers around ``@st.cache_data`` / ``@st.cache_resource`` so that
other modules don't need to import Streamlit directly.
"""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Computing statistics …")
def cached_describe(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cache-friendly describe()."""
    return df.describe(include="all")


def dataframe_hash(df: pd.DataFrame) -> str:
    """Fast, deterministic hash of a DataFrame for cache keys."""
    h = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
    return h
