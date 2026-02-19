"""
🤖 Model Training – preprocessing, model selection, training & evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.session_state import init_session_state, has_data, has_target
from src.ui.sidebar import render_sidebar
from src.ui.model_config import render_model_selector, render_all_hyperparams
from src.data.preprocessing import (
    IMPUTE_STRATEGIES_NUM,
    IMPUTE_STRATEGIES_CAT,
    ENCODING_STRATEGIES,
    SCALING_STRATEGIES,
    preprocess,
)
from src.data.splitter import SPLIT_STRATEGIES, split_data
from src.models.trainer import train_multiple
from src.models.evaluator import evaluate_all

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")
init_session_state()
render_sidebar()

st.title("🤖 Model Training")

if not has_data():
    st.info("⬅️ Load a dataset on the **Home** page first.")
    st.stop()

if not has_target():
    st.info("⬅️ Select a target column on the **Home** page first.")
    st.stop()

df: pd.DataFrame = st.session_state.df
target_col: str = st.session_state.target_col
task: str = st.session_state.task_type

# ═══════════════════════════════════════════════════════════════
# 1 – Preprocessing
# ═══════════════════════════════════════════════════════════════
st.header("1️⃣ Preprocessing")

col1, col2, col3, col4 = st.columns(4)
with col1:
    num_impute = st.selectbox("Numeric imputation", IMPUTE_STRATEGIES_NUM, index=1, key="pp_num")
with col2:
    cat_impute = st.selectbox("Categorical imputation", IMPUTE_STRATEGIES_CAT, index=0, key="pp_cat")
with col3:
    encoding = st.selectbox("Encoding", ENCODING_STRATEGIES, index=0, key="pp_enc")
with col4:
    scaling = st.selectbox("Scaling", SCALING_STRATEGIES, index=0, key="pp_scl")

# ═══════════════════════════════════════════════════════════════
# 2 – Train / Test Split
# ═══════════════════════════════════════════════════════════════
st.header("2️⃣ Train / Test Split")
col_a, col_b, col_c = st.columns(3)
with col_a:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="test_sz")
with col_b:
    split_strat = st.selectbox("Split strategy", SPLIT_STRATEGIES, index=0, key="split_strat")
with col_c:
    random_seed = st.number_input("Random seed", value=42, min_value=0, key="seed")

# ═══════════════════════════════════════════════════════════════
# 3 – Model Selection
# ═══════════════════════════════════════════════════════════════
st.header("3️⃣ Model Selection")
selected_keys = render_model_selector(task)

if not selected_keys:
    st.warning("Select at least one model.")
    st.stop()

params_map = render_all_hyperparams(selected_keys, task)

# ═══════════════════════════════════════════════════════════════
# 4 – Train!
# ═══════════════════════════════════════════════════════════════
st.header("4️⃣ Train Models")

if st.button("🚀 Start Training", type="primary", use_container_width=True):
    with st.spinner("Preprocessing data …"):
        df_processed = preprocess(
            df, target_col,
            num_impute=num_impute,
            cat_impute=cat_impute,
            encoding=encoding,
            scaling=scaling,
        )
        st.session_state.df_processed = df_processed

        X_train, X_test, y_train, y_test = split_data(
            df_processed, target_col,
            test_size=test_size,
            strategy=split_strat,
            random_state=int(random_seed),
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    st.success(f"✅ Data preprocessed — Train: {len(X_train)}, Test: {len(X_test)}")

    # Progress bar
    progress = st.progress(0)
    status_text = st.empty()

    def _progress_cb(i: int, total: int, name: str):
        progress.progress(i / total)
        status_text.text(f"Training {name} ({i}/{total}) …")

    with st.spinner("Training models …"):
        train_results = train_multiple(
            selected_keys, X_train, y_train,
            params_map=params_map,
            progress_callback=_progress_cb,
        )
        st.session_state.train_results = train_results

    progress.progress(1.0)
    status_text.text("Evaluating …")

    eval_results = evaluate_all(train_results, X_test, y_test)
    st.session_state.eval_results = eval_results

    status_text.empty()
    progress.empty()

    st.success(f"🎉 Trained & evaluated **{len(eval_results)}** models!")
    st.balloons()

    # Quick peek
    from src.models.evaluator import comparison_dataframe
    comp = comparison_dataframe(eval_results)
    st.dataframe(
        comp.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
        use_container_width=True,
    )
    st.info("Head to the **📈 Results** page for detailed comparisons.")
