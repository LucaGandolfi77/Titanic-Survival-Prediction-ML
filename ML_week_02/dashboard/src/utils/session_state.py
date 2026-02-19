"""
session_state.py – Manage Streamlit session state for XAI dashboard.

Centralises all state keys and provides typed accessors / defaults.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

# Default values for every session key
_DEFAULTS: Dict[str, Any] = {
    # Data
    "dataset_name": None,
    "dataframe": None,
    "target_col": None,
    "feature_cols": None,
    "protected_attrs": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,

    # Model
    "model": None,
    "model_name": None,
    "model_info": None,
    "predictions": None,
    "performance": None,

    # Explainability
    "shap_values": None,
    "shap_explainer": None,
    "expected_value": None,
    "lime_explainer": None,

    # Fairness
    "fairness_results": None,
    "bias_scan": None,
    "recommendations": None,

    # Reports
    "executive_summary": None,
    "technical_report": None,
}


def init_state() -> None:
    """Initialise all session-state keys with defaults (idempotent)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str) -> Any:
    """Get a session-state value (returns None if unset)."""
    return st.session_state.get(key, _DEFAULTS.get(key))


def put(key: str, value: Any) -> None:
    """Set a session-state value."""
    st.session_state[key] = value


def put_many(**kwargs: Any) -> None:
    """Set multiple state values at once."""
    for k, v in kwargs.items():
        st.session_state[k] = v


def clear_downstream(from_stage: str = "data") -> None:
    """Clear all state downstream of a given stage.

    Stages (in order): data → model → explain → fairness → report
    """
    stages = {
        "data": [
            "dataframe", "target_col", "feature_cols", "protected_attrs",
            "X_train", "X_test", "y_train", "y_test",
            "model", "model_name", "model_info", "predictions", "performance",
            "shap_values", "shap_explainer", "expected_value", "lime_explainer",
            "fairness_results", "bias_scan", "recommendations",
            "executive_summary", "technical_report",
        ],
        "model": [
            "model", "model_name", "model_info", "predictions", "performance",
            "shap_values", "shap_explainer", "expected_value", "lime_explainer",
            "fairness_results", "bias_scan", "recommendations",
            "executive_summary", "technical_report",
        ],
        "explain": [
            "shap_values", "shap_explainer", "expected_value", "lime_explainer",
            "fairness_results", "bias_scan", "recommendations",
            "executive_summary", "technical_report",
        ],
        "fairness": [
            "fairness_results", "bias_scan", "recommendations",
            "executive_summary", "technical_report",
        ],
        "report": [
            "executive_summary", "technical_report",
        ],
    }

    for key in stages.get(from_stage, []):
        st.session_state[key] = _DEFAULTS.get(key)


def has(key: str) -> bool:
    """Check if a key is set and not None."""
    return st.session_state.get(key) is not None


def stage_ready(stage: str) -> bool:
    """Check if a pipeline stage has completed."""
    checks = {
        "data": ["dataframe", "target_col"],
        "model": ["model", "performance"],
        "explain": ["shap_values"],
        "fairness": ["fairness_results"],
        "report": ["executive_summary"],
    }
    return all(has(k) for k in checks.get(stage, []))
