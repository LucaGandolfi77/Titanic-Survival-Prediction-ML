"""
1_overview.py – Data loading, model training / loading, and pipeline overview.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.models.loader import list_available_models, load_model, load_metadata
from src.models.metadata import extract_model_info, compute_performance
from src.models.predictor import Predictor
from src.utils.data_loader import (
    list_available_datasets, load_csv, load_uploaded_file,
    get_dataset_path, dataset_summary, prepare_xy,
)
from src.utils.helpers import auto_detect_target, auto_detect_protected, load_config
from src.utils.session_state import put, put_many, get, has, clear_downstream, stage_ready
from src.visualization.custom_plots import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_distribution,
    plot_correlation_matrix, plot_missing_values, plot_model_radar,
)

st.header("🏠 Overview – Data & Model Setup")

# ══════════════════════════════════════════════════════════════
# STEP 1 – Dataset
# ══════════════════════════════════════════════════════════════
st.subheader("1️⃣ Load Dataset")

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    datasets = list_available_datasets()
    selected_ds = st.selectbox("Built-in dataset", ["—"] + datasets)

with col_ds2:
    uploaded = st.file_uploader("Or upload your own (CSV/Excel)", type=["csv", "xlsx", "xls"])

# Load
if uploaded is not None:
    df = load_uploaded_file(uploaded)
    put("dataset_name", uploaded.name)
    put("dataframe", df)
elif selected_ds != "—":
    df = load_csv(get_dataset_path(selected_ds))
    put("dataset_name", selected_ds)
    put("dataframe", df)

df = get("dataframe")
if df is None:
    st.info("👆 Load a dataset to get started.")
    st.stop()

# ── Dataset overview ──────────────────────────────────────────
stats = dataset_summary(df)
st.markdown(f"**{get('dataset_name')}** — {stats['rows']:,} rows × {stats['columns']} cols "
            f"| {stats['numeric_cols']} numeric | {stats['categorical_cols']} categorical "
            f"| {stats['missing_pct']} missing | {stats['memory_mb']} MB")

with st.expander("Preview & Stats", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Head", "Describe", "Missing"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)
    with tab3:
        fig = plot_missing_values(df)
        st.plotly_chart(fig, use_container_width=True)

# ── Column selection ──────────────────────────────────────────
st.subheader("2️⃣ Configure Columns")
col_a, col_b = st.columns(2)
with col_a:
    default_target = auto_detect_target(df)
    target = st.selectbox("Target column", df.columns.tolist(),
                          index=df.columns.tolist().index(default_target) if default_target in df.columns else 0)
    put("target_col", target)

with col_b:
    default_protected = auto_detect_protected(df)
    protected = st.multiselect("Protected attributes (for fairness)",
                               [c for c in df.columns if c != target],
                               default=[p for p in default_protected if p != target and p in df.columns])
    put("protected_attrs", protected)

feature_cols = [c for c in df.columns if c != target]
put("feature_cols", feature_cols)

# ── Correlation matrix ────────────────────────────────────────
with st.expander("Feature Correlations"):
    fig_corr = plot_correlation_matrix(df)
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# STEP 2 – Model
# ══════════════════════════════════════════════════════════════
st.subheader("3️⃣ Model Setup")

model_tab1, model_tab2 = st.tabs(["🗂️ Load Pre-trained", "🛠️ Train Quick Model"])

# ── Load pre-trained model ────────────────────────────────────
with model_tab1:
    available_models = list_available_models()
    if available_models:
        selected_model = st.selectbox("Select model", available_models)
        if st.button("Load Model", key="load_model_btn"):
            with st.spinner("Loading…"):
                model = load_model(selected_model)
                predictor = Predictor(model)
                info = extract_model_info(model, feature_names=feature_cols)
                put_many(model=model, model_name=selected_model, model_info=info)

                metadata = load_metadata(selected_model)
                if metadata:
                    put("performance", metadata.get("performance", {}))

                st.success(f"✅ Loaded **{selected_model}**")
    else:
        st.info("No pre-trained models found in `models/`. Train one below or add `.pkl` files.")

# ── Train quick model ─────────────────────────────────────────
with model_tab2:
    algo = st.selectbox("Algorithm", [
        "Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost", "LightGBM",
    ])

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)
    with col_p2:
        random_state = st.number_input("Random seed", value=42, step=1)
    with col_p3:
        n_estimators = st.number_input("Estimators (tree models)", value=100, step=50, min_value=10)

    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner(f"Training {algo}…"):
            X, y = prepare_xy(df, target, feature_cols)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=y if y.nunique() <= 20 else None
            )

            # Build model
            if algo == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
            elif algo == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=int(n_estimators), random_state=int(random_state))
            elif algo == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=int(random_state))
            elif algo == "XGBoost":
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=int(n_estimators), random_state=int(random_state),
                                      use_label_encoder=False, eval_metric="logloss")
            else:  # LightGBM
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=int(n_estimators), random_state=int(random_state), verbose=-1)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            info = extract_model_info(model, feature_names=list(X.columns))
            perf = compute_performance(y_test, y_pred, model, X_test)

            put_many(
                model=model, model_name=algo, model_info=info,
                predictions=y_pred, performance=perf,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            )
            st.success(f"✅ **{algo}** trained — Accuracy: {perf.get('accuracy', 0):.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 3 – Model overview
# ══════════════════════════════════════════════════════════════
if has("model") and has("performance"):
    st.subheader("4️⃣ Model Performance Overview")

    perf = get("performance")
    cols = st.columns(min(len(perf), 5))
    for i, (k, v) in enumerate(list(perf.items())[:5]):
        with cols[i]:
            st.metric(k.replace("_", " ").title(), f"{v:.4f}")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        radar_metrics = {k: v for k, v in perf.items() if isinstance(v, (int, float)) and 0 <= v <= 1}
        if radar_metrics:
            fig_radar = plot_model_radar(radar_metrics)
            st.plotly_chart(fig_radar, use_container_width=True)

    with col_r2:
        cm = perf.get("confusion_matrix")
        if cm is not None:
            fig_cm = plot_confusion_matrix(np.array(cm))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.success("✅ Data & Model ready — proceed to **Global Explanations** →")
