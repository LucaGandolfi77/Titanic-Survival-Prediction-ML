"""
export.py – Model & prediction download helpers.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model serialisation ──────────────────────────────────────

def save_model(estimator: BaseEstimator, name: str) -> Path:
    """Persist a trained model as .pkl and return the path."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"{name}_{ts}.pkl"
    joblib.dump(estimator, path)
    return path


def model_download_button(
    estimator: BaseEstimator,
    display_name: str,
    key: str = "dl_model",
) -> None:
    """Streamlit download button for a pickled model."""
    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    buf.seek(0)
    st.download_button(
        label=f"⬇️ Download {display_name} (.pkl)",
        data=buf,
        file_name=f"{display_name.replace(' ', '_').lower()}.pkl",
        mime="application/octet-stream",
        key=key,
    )


# ── Predictions ───────────────────────────────────────────────

def save_predictions(
    y_pred: pd.Series | pd.DataFrame,
    name: str,
) -> Path:
    """Save predictions to CSV and return the path."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PREDICTIONS_DIR / f"{name}_predictions_{ts}.csv"
    pd.DataFrame(y_pred).to_csv(path, index=False)
    return path


def predictions_download_button(
    y_pred,
    display_name: str,
    key: str = "dl_preds",
) -> None:
    """Streamlit download button for predictions CSV."""
    csv = pd.DataFrame(y_pred, columns=["prediction"]).to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download Predictions – {display_name}",
        data=csv,
        file_name=f"{display_name.replace(' ', '_').lower()}_predictions.csv",
        mime="text/csv",
        key=key,
    )


# ── Report ────────────────────────────────────────────────────

def save_report(content: str, name: str) -> Path:
    """Save a text report."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{name}_report_{ts}.txt"
    path.write_text(content)
    return path
