"""
loader.py – CSV / Excel upload, validation & sample-dataset loading.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "assets" / "sample_datasets"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}


# ── public helpers ────────────────────────────────────────────

def load_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame.

    Supports CSV, TSV and Excel formats.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {Path(name).suffix}")
    except Exception as exc:
        raise ValueError(f"Could not parse file: {exc}") from exc

    return _validate(df)


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load one of the bundled sample CSVs by filename (e.g. 'iris.csv')."""
    path = SAMPLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Sample dataset not found: {path}")
    df = pd.read_csv(path)
    return _validate(df)


def list_sample_datasets() -> list[str]:
    """Return sorted list of available sample dataset filenames."""
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLE_DIR.iterdir() if p.suffix in SUPPORTED_EXTENSIONS)


# ── validation ────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity checks after loading."""
    if df.empty:
        raise ValueError("The uploaded file is empty.")
    if df.shape[1] < 2:
        raise ValueError("The dataset must have at least 2 columns.")
    # Drop completely empty rows / columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # Reset index after dropping
    df = df.reset_index(drop=True)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Quick summary dict for the sidebar."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_cols": df.select_dtypes("number").columns.tolist(),
        "categorical_cols": df.select_dtypes(["object", "category"]).columns.tolist(),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicated_rows": int(df.duplicated().sum()),
    }
