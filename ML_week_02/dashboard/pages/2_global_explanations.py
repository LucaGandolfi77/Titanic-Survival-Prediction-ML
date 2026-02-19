"""
2_global_explanations.py – SHAP-based global model interpretability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.pdp import compute_pdp, plot_pdp, compute_pdp_2d, plot_pdp_2d
from src.explainability.feature_interactions import top_interactions, plot_interaction_heatmap
from src.utils.session_state import get, put, put_many, has, stage_ready
from src.visualization.shap_plots import (
    plot_global_importance, plot_beeswarm, plot_dependence,
)

st.header("🌐 Global Explanations")

# ── Guard: need data + model ──────────────────────────────────
if not stage_ready("model"):
    st.warning("⚠️ Please load data and a model on the **Overview** page first.")
    st.stop()

model = get("model")
X_test = get("X_test")
X_train = get("X_train")
feature_names = list(X_test.columns) if X_test is not None else get("feature_cols")

if X_test is None or X_train is None:
    st.warning("⚠️ Train/test data not available. Please train a model on the Overview page.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# Compute SHAP values
# ══════════════════════════════════════════════════════════════
st.subheader("1️⃣ SHAP Analysis")

max_samples = st.slider("Background samples (speed ↔ accuracy)", 50, 500, 100, 50)

if st.button("🧠 Compute SHAP Values", key="compute_shap") or has("shap_values"):
    if not has("shap_values"):
        with st.spinner("Computing SHAP values … this may take a moment"):
            explainer = SHAPExplainer(model, X_train.sample(min(max_samples, len(X_train))))
            shap_vals = explainer.compute_shap_values(X_test)
            importance = explainer.global_importance(X_test)
            put_many(
                shap_values=shap_vals,
                shap_explainer=explainer,
                expected_value=explainer.explainer.expected_value,
            )
            st.success("✅ SHAP values computed!")

    shap_vals = get("shap_values")
    expected_val = get("expected_value")

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Importance", "🐝 Beeswarm", "📈 Dependence", "🔗 Interactions",
    ])

    # Feature importance bar chart
    with tab1:
        top_k = st.slider("Top features", 5, min(30, len(feature_names)), 15, key="imp_topk")
        fig = plot_global_importance(shap_vals, feature_names, top_k=top_k)
        st.plotly_chart(fig, use_container_width=True)

        # Downloadable table
        explainer_obj = get("shap_explainer")
        if explainer_obj:
            imp_df = explainer_obj.global_importance(X_test)
            st.dataframe(imp_df.head(top_k), use_container_width=True)

    # Beeswarm
    with tab2:
        top_k_bee = st.slider("Top features", 5, min(25, len(feature_names)), 12, key="bee_topk")
        fig_bee = plot_beeswarm(shap_vals, X_test, top_k=top_k_bee)
        st.plotly_chart(fig_bee, use_container_width=True)

    # Dependence plots
    with tab3:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dep_feature = st.selectbox("Feature", feature_names, key="dep_feat")
        with col_d2:
            int_feature = st.selectbox("Color by (interaction)", ["None"] + feature_names, key="dep_int")
        int_feat = int_feature if int_feature != "None" else None
        fig_dep = plot_dependence(shap_vals, X_test, dep_feature, int_feat)
        st.plotly_chart(fig_dep, use_container_width=True)

        # PDP overlay
        with st.expander("Partial Dependence Plot"):
            pdp_data = compute_pdp(model, X_test, dep_feature)
            fig_pdp = plot_pdp(pdp_data, dep_feature)
            st.plotly_chart(fig_pdp, use_container_width=True)

    # Feature interactions
    with tab4:
        st.markdown("**Top Feature Interactions (H-statistic)**")
        n_top = st.slider("Top pairs", 3, 15, 8, key="int_topn")
        with st.spinner("Computing interactions…"):
            interactions = top_interactions(model, X_test.sample(min(200, len(X_test))),
                                            top_n=n_top, features=feature_names)
        if not interactions.empty:
            st.dataframe(interactions, use_container_width=True)
            fig_h = plot_interaction_heatmap(interactions)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Not enough features for interaction analysis.")

    st.success("✅ Global explanations ready — proceed to **Local Explanations** →")
