"""
helpers.py – Miscellaneous utility functions for the XAI dashboard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


# ── Config loading ────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dashboard_config.yaml"


def load_config() -> Dict[str, Any]:
    """Load the dashboard YAML configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_protected_attributes() -> Dict[str, Any]:
    """Load protected attribute definitions."""
    path = Path(__file__).resolve().parents[2] / "data" / "protected_attributes.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ── Numeric helpers ───────────────────────────────────────────

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns *default* instead of raising ZeroDivisionError."""
    return a / b if b != 0 else default


def percentile_clip(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Clip array to [lo, hi] percentiles."""
    low = np.percentile(arr, lo)
    high = np.percentile(arr, hi)
    return np.clip(arr, low, high)


# ── Dataframe helpers ─────────────────────────────────────────

def auto_detect_target(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: detect likely target column.

    Looks for columns named 'target', 'label', 'class', 'y' or columns
    with very few unique values.
    """
    common_names = {"target", "label", "class", "y", "outcome", "default",
                    "survived", "churn", "fraud", "diagnosis", "approved"}
    for col in df.columns:
        if col.lower().strip() in common_names:
            return col

    # Fallback: last column with few unique values
    for col in reversed(df.columns):
        if df[col].nunique() <= 10:
            return col
    return df.columns[-1]


def auto_detect_protected(
    df: pd.DataFrame,
    known_attrs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Heuristic: detect likely protected / sensitive columns."""
    suspects = {"sex", "gender", "race", "ethnicity", "age", "age_group",
                "religion", "disability", "marital_status", "nationality",
                "country", "skin_color"}
    found = []
    for col in df.columns:
        if col.lower().strip() in suspects:
            found.append(col)
    if known_attrs:
        for attr in known_attrs:
            if attr in df.columns and attr not in found:
                found.append(attr)
    return found


# ── Model type detection ─────────────────────────────────────

def model_family(model: Any) -> str:
    """Return a human-readable model family string."""
    name = type(model).__name__
    families = {
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "XGB": "XGBoost",
        "LGBM": "LightGBM",
        "CatBoost": "CatBoost",
        "LogisticRegression": "Logistic Regression",
        "SVC": "SVM",
        "DecisionTree": "Decision Tree",
        "KNeighbors": "k-NN",
        "MLP": "Neural Network",
        "AdaBoost": "AdaBoost",
    }
    for key, family in families.items():
        if key in name:
            return family
    return name


# ── File hashing (for cache invalidation) ─────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file (first 16 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Colour helpers for UI ─────────────────────────────────────

STATUS_COLOURS = {
    "PASS": "#5CB85C",
    "WARNING": "#F5A623",
    "FAIL": "#D9534F",
}


def status_color(status: str) -> str:
    """Map a status string to a hex colour."""
    for key, colour in STATUS_COLOURS.items():
        if key in status.upper():
            return colour
    return "#888888"


def status_emoji(status: str) -> str:
    """Map status to emoji."""
    if "PASS" in status.upper():
        return "✅"
    if "WARN" in status.upper():
        return "⚠️"
    if "FAIL" in status.upper():
        return "❌"
    return "ℹ️"
