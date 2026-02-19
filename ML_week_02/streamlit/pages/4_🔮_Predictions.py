"""
🔮 Predictions – Run inference with a trained model on new data.
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

from src.utils.session_state import init_session_state, has_results
from src.ui.sidebar import render_sidebar
from src.data.loader import load_uploaded_file
from src.data.preprocessing import preprocess
from src.utils.export import predictions_download_button

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
init_session_state()
render_sidebar()

st.title("🔮 Predictions")

if not has_results():
    st.info("⬅️ Train models on the **🤖 Model Training** page first.")
    st.stop()

eval_results = st.session_state.eval_results
train_results = st.session_state.train_results
target_col = st.session_state.target_col
task = st.session_state.task_type

# ═══════════════════════════════════════════════════════════════
# Model choice
# ═══════════════════════════════════════════════════════════════
st.subheader("1️⃣ Select a trained model")
model_names = [er.display_name for er in eval_results]
chosen = st.selectbox("Model", model_names, key="pred_model")
chosen_tr = next(tr for tr in train_results if tr.display_name == chosen)
chosen_er = next(er for er in eval_results if er.display_name == chosen)

st.info(f"**{chosen}** — primary metric: **{list(chosen_er.metrics.values())[0]:.4f}**")

# ═══════════════════════════════════════════════════════════════
# Upload new data
# ═══════════════════════════════════════════════════════════════
st.subheader("2️⃣ Upload new data for prediction")
st.caption("The file should have the **same features** as the training data (target column is optional).")

uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "tsv"], key="pred_upload")

if uploaded is not None:
    try:
        new_df = load_uploaded_file(uploaded)
        st.success(f"✅ Loaded {new_df.shape[0]} rows × {new_df.shape[1]} columns")
        st.dataframe(new_df.head(20), use_container_width=True)

        # Pre-process the same way as training
        with st.spinner("Preprocessing …"):
            # Drop target if present
            if target_col in new_df.columns:
                new_df = new_df.drop(columns=[target_col])

            new_df_processed = preprocess(
                new_df, target_col="__dummy__",
                num_impute=st.session_state.get("num_impute", "median"),
                cat_impute=st.session_state.get("cat_impute", "most_frequent"),
                encoding=st.session_state.get("encoding", "one_hot"),
                scaling=st.session_state.get("scaling", "standard"),
            )

            # Align columns with training data
            train_cols = st.session_state.X_train.columns.tolist()
            for c in train_cols:
                if c not in new_df_processed.columns:
                    new_df_processed[c] = 0
            new_df_processed = new_df_processed[train_cols]

        st.subheader("3️⃣ Predictions")
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            preds = chosen_tr.estimator.predict(new_df_processed)

            result_df = new_df.copy()
            result_df["prediction"] = preds

            if hasattr(chosen_tr.estimator, "predict_proba"):
                proba = chosen_tr.estimator.predict_proba(new_df_processed)
                for i, cls in enumerate(chosen_tr.estimator.classes_):
                    result_df[f"prob_{cls}"] = proba[:, i].round(4)

            st.dataframe(result_df, use_container_width=True)

            # Download
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"❌ Error: {exc}")

# ═══════════════════════════════════════════════════════════════
# Manual single-row prediction
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📝 Quick Single-Row Prediction")
st.caption("Fill in feature values manually.")

if st.session_state.X_train is not None:
    feature_cols = st.session_state.X_train.columns.tolist()
    input_data = {}

    cols = st.columns(min(len(feature_cols), 4))
    for i, feat in enumerate(feature_cols):
        with cols[i % len(cols)]:
            default_val = float(st.session_state.X_train[feat].median())
            input_data[feat] = st.number_input(feat, value=default_val, key=f"manual_{feat}")

    if st.button("🔮 Predict (single row)", key="single_pred"):
        row = pd.DataFrame([input_data])
        pred = chosen_tr.estimator.predict(row)
        st.success(f"**Prediction:** {pred[0]}")

        if hasattr(chosen_tr.estimator, "predict_proba"):
            proba = chosen_tr.estimator.predict_proba(row)[0]
            for cls, p in zip(chosen_tr.estimator.classes_, proba):
                st.metric(f"P(class={cls})", f"{p:.4f}")
