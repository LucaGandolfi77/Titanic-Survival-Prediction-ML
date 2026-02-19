"""
sidebar.py – Shared sidebar components.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.loader import get_dataset_summary


LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "logo.png"


def render_sidebar() -> None:
    """Render the global sidebar: logo, dataset summary, navigation hints."""
    with st.sidebar:
        st.markdown("## 🧪 ML Playground")
        st.caption("Interactive Machine Learning Dashboard")
        st.markdown("---")

        # Dataset quick info
        if "df" in st.session_state and st.session_state.df is not None:
            summary = get_dataset_summary(st.session_state.df)
            st.markdown("### 📋 Dataset")
            col1, col2 = st.columns(2)
            col1.metric("Rows", summary["rows"])
            col2.metric("Columns", summary["columns"])
            col1.metric("Numeric", len(summary["numeric_cols"]))
            col2.metric("Categorical", len(summary["categorical_cols"]))
            st.metric("Missing cells", f"{summary['missing_cells']} ({summary['missing_pct']}%)")
            st.markdown("---")

            # Target column (if set)
            if "target_col" in st.session_state:
                st.info(f"**Target:** `{st.session_state.target_col}`")
                st.info(f"**Task:** `{st.session_state.get('task_type', 'auto')}`")
        else:
            st.info("Upload a dataset or choose a sample to begin.")

        st.markdown("---")
        st.caption("Built with Streamlit • scikit-learn • Plotly")
