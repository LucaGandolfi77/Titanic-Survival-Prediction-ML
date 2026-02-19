"""
data_upload.py – File-upload & sample-dataset selection UI.
"""
from __future__ import annotations

import streamlit as st

from src.data.loader import list_sample_datasets, load_sample_dataset, load_uploaded_file


def render_upload_section() -> None:
    """Render the data upload / sample-selection section.

    Stores the loaded DataFrame in ``st.session_state.df``.
    """
    st.header("📂 Load Dataset")

    tab_upload, tab_sample = st.tabs(["Upload File", "Sample Dataset"])

    # ── Upload tab ────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Maximum file size: 200 MB",
        )
        if uploaded is not None:
            try:
                df = load_uploaded_file(uploaded)
                st.session_state.df = df
                st.session_state.dataset_name = uploaded.name
                st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
            except ValueError as exc:
                st.error(f"❌ {exc}")

    # ── Sample tab ────────────────────────────────────────────
    with tab_sample:
        samples = list_sample_datasets()
        if not samples:
            st.warning("No sample datasets found in `assets/sample_datasets/`.")
        else:
            choice = st.selectbox("Pick a sample dataset", samples)
            if st.button("Load sample", type="primary"):
                try:
                    df = load_sample_dataset(choice)
                    st.session_state.df = df
                    st.session_state.dataset_name = choice
                    st.success(f"✅ Loaded **{choice}** — {df.shape[0]} rows × {df.shape[1]} columns")
                except Exception as exc:
                    st.error(f"❌ {exc}")


def render_target_selector() -> None:
    """Let the user pick the target column & auto-detect task type."""
    if "df" not in st.session_state or st.session_state.df is None:
        return

    df = st.session_state.df
    st.subheader("🎯 Select Target Column")

    target = st.selectbox(
        "Target column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,  # default: last column
        help="The column you want to predict.",
    )
    st.session_state.target_col = target

    # Auto-detect task type
    nunique = df[target].nunique()
    is_numeric = df[target].dtype.kind in "iufb"

    if is_numeric and nunique > 20:
        default_task = "regression"
    else:
        default_task = "classification"

    task = st.radio(
        "Task type",
        ["classification", "regression"],
        index=0 if default_task == "classification" else 1,
        horizontal=True,
    )
    st.session_state.task_type = task
