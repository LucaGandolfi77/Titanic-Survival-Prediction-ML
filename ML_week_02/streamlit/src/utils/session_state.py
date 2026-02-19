"""
session_state.py – Centralised Streamlit session-state management.

Call ``init_session_state()`` once at app startup to ensure every key
used throughout the app has a safe default.
"""
from __future__ import annotations

import streamlit as st


_DEFAULTS = {
    # Data
    "df": None,
    "df_processed": None,
    "dataset_name": None,
    "target_col": None,
    "task_type": None,
    # Splits
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    # Preprocessing settings
    "num_impute": "median",
    "cat_impute": "most_frequent",
    "encoding": "one_hot",
    "scaling": "standard",
    "test_size": 0.2,
    "split_strategy": "holdout",
    # Training results
    "train_results": [],
    "eval_results": [],
    # Misc
    "selected_models": [],
    "params_map": {},
}


def init_session_state() -> None:
    """Initialise every session-state key with its default if missing."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_session_state() -> None:
    """Reset all managed keys back to defaults."""
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default


def has_data() -> bool:
    return st.session_state.get("df") is not None


def has_target() -> bool:
    return st.session_state.get("target_col") is not None


def has_splits() -> bool:
    return st.session_state.get("X_train") is not None


def has_results() -> bool:
    return bool(st.session_state.get("eval_results"))
