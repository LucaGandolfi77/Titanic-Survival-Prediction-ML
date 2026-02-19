"""
📊 Data Explorer – EDA page.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from src.utils.session_state import init_session_state, has_data
from src.ui.sidebar import render_sidebar
from src.eda.statistics import (
    descriptive_stats,
    numeric_summary,
    categorical_summary,
    target_distribution,
)
from src.eda.distributions import (
    histogram,
    box_plot,
    all_numeric_histograms,
    violin_plot,
)
from src.eda.correlations import (
    correlation_heatmap,
    top_correlations,
    scatter_matrix,
)
from src.eda.outliers import outlier_summary, outlier_box_plot

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
init_session_state()
render_sidebar()

st.title("📊 Data Explorer")

if not has_data():
    st.info("⬅️ Go to the **Home** page to load a dataset first.")
    st.stop()

df: pd.DataFrame = st.session_state.df

# ── Tabs ──────────────────────────────────────────────────────
tab_overview, tab_dist, tab_corr, tab_outlier = st.tabs(
    ["Overview", "Distributions", "Correlations", "Outliers"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 – Overview
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(descriptive_stats(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeric Summary")
        ns = numeric_summary(df)
        if not ns.empty:
            st.dataframe(ns, use_container_width=True)
        else:
            st.caption("No numeric columns found.")
    with col2:
        st.subheader("Categorical Summary")
        cs = categorical_summary(df)
        if not cs.empty:
            st.dataframe(cs, use_container_width=True)
        else:
            st.caption("No categorical columns found.")

    # Target distribution
    if st.session_state.get("target_col"):
        st.subheader("Target Distribution")
        td = target_distribution(df, st.session_state.target_col)
        st.dataframe(td, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 – Distributions
# ═══════════════════════════════════════════════════════════════
with tab_dist:
    st.subheader("Feature Distributions")

    # All numeric histograms
    fig_all = all_numeric_histograms(df)
    if fig_all.data:
        st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("---")
    st.subheader("Single Feature Explorer")
    col_sel = st.selectbox("Select feature", df.columns.tolist(), key="dist_col")
    color_by = st.selectbox(
        "Color by (optional)",
        [None] + df.select_dtypes(["object", "category"]).columns.tolist()
        + ([st.session_state.target_col] if st.session_state.get("target_col") else []),
        key="dist_color",
    )

    chart_type = st.radio("Chart type", ["Histogram", "Box plot", "Violin"], horizontal=True, key="dist_type")

    if chart_type == "Histogram":
        st.plotly_chart(histogram(df, col_sel, color=color_by), use_container_width=True)
    elif chart_type == "Box plot":
        st.plotly_chart(box_plot(df, col_sel, group_by=color_by), use_container_width=True)
    else:
        st.plotly_chart(violin_plot(df, col_sel, group_by=color_by), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 – Correlations
# ═══════════════════════════════════════════════════════════════
with tab_corr:
    st.subheader("Correlation Analysis")

    method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True, key="corr_method")

    st.plotly_chart(correlation_heatmap(df, method=method), use_container_width=True)

    st.subheader("Top Correlations")
    target = st.session_state.get("target_col")
    top = top_correlations(df, target_col=target, method=method, top_n=15)
    st.dataframe(top, use_container_width=True)

    st.subheader("Scatter Matrix")
    num_cols = df.select_dtypes("number").columns.tolist()
    selected = st.multiselect("Columns for scatter matrix", num_cols, default=num_cols[:4], key="scatter_cols")
    if len(selected) >= 2:
        st.plotly_chart(
            scatter_matrix(df, columns=selected, color=target),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 4 – Outliers
# ═══════════════════════════════════════════════════════════════
with tab_outlier:
    st.subheader("Outlier Detection")

    method_out = st.radio("Detection method", ["iqr", "zscore"], horizontal=True, key="outlier_method")

    if method_out == "iqr":
        factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1, key="iqr_factor")
        summary = outlier_summary(df, method="iqr", factor=factor)
    else:
        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh")
        summary = outlier_summary(df, method="zscore", threshold=threshold)

    st.dataframe(summary, use_container_width=True)

    st.plotly_chart(outlier_box_plot(df), use_container_width=True)
