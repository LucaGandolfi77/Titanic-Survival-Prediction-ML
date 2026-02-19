"""
3_local_explanations.py – Instance-level SHAP + LIME explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.counterfactuals import what_if_analysis, find_counterfactuals
from src.utils.session_state import get, put, has, stage_ready
from src.visualization.shap_plots import plot_waterfall, plot_force_horizontal
from src.visualization.lime_plots import plot_lime_explanation, plot_shap_vs_lime

st.header("🔬 Local Explanations")

# ── Guard ─────────────────────────────────────────────────────
if not stage_ready("model"):
    st.warning("⚠️ Please load data and a model on the **Overview** page first.")
    st.stop()

model = get("model")
X_test = get("X_test")
X_train = get("X_train")
y_test = get("y_test")
shap_vals = get("shap_values")
expected_val = get("expected_value")
feature_names = list(X_test.columns) if X_test is not None else get("feature_cols")

if X_test is None:
    st.warning("⚠️ No test data available.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# Instance selection
# ══════════════════════════════════════════════════════════════
st.subheader("1️⃣ Select Instance")

col_sel1, col_sel2 = st.columns([3, 1])
with col_sel1:
    idx = st.number_input("Row index", 0, len(X_test) - 1, 0, key="local_idx")
with col_sel2:
    if st.button("🎲 Random"):
        idx = int(np.random.randint(0, len(X_test)))
        st.rerun()

instance = X_test.iloc[[idx]]
st.dataframe(instance, use_container_width=True)

# Show prediction
try:
    pred = model.predict(instance)[0]
    proba = model.predict_proba(instance)[0] if hasattr(model, "predict_proba") else None
    st.markdown(f"**Prediction:** `{pred}`" + (f" — Probability: `{proba}`" if proba is not None else ""))
except Exception as e:
    st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════
# Tabs: SHAP | LIME | What-If | Comparison
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🧊 SHAP", "🍋 LIME", "🔮 What-If", "⚖️ Compare"])

# ── SHAP local ────────────────────────────────────────────────
with tab1:
    if shap_vals is not None:
        ev = expected_val
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
        ev = float(ev)

        st.markdown("**Waterfall Plot**")
        fig_wf = plot_waterfall(shap_vals, feature_names, ev, instance_idx=idx)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("**Force Plot**")
        fig_fp = plot_force_horizontal(shap_vals, feature_names, ev, instance_idx=idx)
        st.plotly_chart(fig_fp, use_container_width=True)

        # Feature contribution table
        sv = shap_vals[idx]
        if sv.ndim > 1:
            sv = sv[:, 1]
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": instance.values[0],
            "SHAP": sv,
        }).sort_values("SHAP", key=abs, ascending=False)
        st.dataframe(contrib_df.head(15), use_container_width=True)
    else:
        st.info("Compute SHAP values on the **Global Explanations** page first.")

# ── LIME local ────────────────────────────────────────────────
with tab2:
    n_features = st.slider("LIME features", 5, 20, 10, key="lime_nf")
    if st.button("🍋 Compute LIME Explanation", key="lime_btn") or has("lime_explainer"):
        with st.spinner("Computing LIME explanation…"):
            lime_exp = LIMEExplainer(model, X_train)
            put("lime_explainer", lime_exp)

            result = lime_exp.explain_instance(
                instance.values[0],
                n_features=n_features,
            )
            fig_lime = plot_lime_explanation(
                result.feature_weights,
                result.prediction,
                result.prediction_proba,
                top_k=n_features,
            )
            st.plotly_chart(fig_lime, use_container_width=True)

            # Table
            st.dataframe(result.feature_weights, use_container_width=True)

# ── What-If analysis ─────────────────────────────────────────
with tab3:
    st.markdown("**Feature Sensitivity (What-If)**")
    wi_feature = st.selectbox("Feature to vary", feature_names, key="wi_feat")
    wi_steps = st.slider("Steps", 10, 100, 30, key="wi_steps")

    if st.button("🔮 Run What-If", key="wi_btn"):
        with st.spinner("Running what-if analysis…"):
            # Generate a range of values spanning the training distribution
            feat_min = float(X_train[wi_feature].min())
            feat_max = float(X_train[wi_feature].max())
            values = np.linspace(feat_min, feat_max, wi_steps)
            wi_df = what_if_analysis(model, instance, wi_feature, values)
            import plotly.express as px
            y_col = "prob_class_1" if "prob_class_1" in wi_df.columns else "prediction"
            fig_wi = px.line(wi_df, x="feature_value", y=y_col,
                             title=f"What-If: varying {wi_feature}",
                             markers=True)
            fig_wi.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_wi, use_container_width=True)

    st.markdown("**Counterfactual Search**")
    if st.button("🔍 Find Counterfactuals", key="cf_btn"):
        with st.spinner("Searching…"):
            # Determine desired class (flip from current prediction)
            current_pred = model.predict(instance)[0]
            classes = model.classes_ if hasattr(model, "classes_") else [0, 1]
            desired = [c for c in classes if c != current_pred]
            desired_class = desired[0] if desired else classes[0]
            cfs = find_counterfactuals(model, instance, desired_class, X_train, n_counterfactuals=5)
            if not cfs.empty:
                st.dataframe(cfs, use_container_width=True)
            else:
                st.info("No counterfactuals found.")

# ── SHAP vs LIME comparison ──────────────────────────────────
with tab4:
    if shap_vals is not None and has("lime_explainer"):
        st.markdown("**SHAP vs LIME Feature Attribution**")

        # SHAP contributions
        sv = shap_vals[idx]
        if sv.ndim > 1:
            sv = sv[:, 1]
        shap_dict = dict(zip(feature_names, sv))

        # LIME contributions (re-compute)
        lime_exp = get("lime_explainer")
        lr = lime_exp.explain_instance(instance.values[0], n_features=15)
        lime_dict = {}
        for _, row in lr.feature_weights.iterrows():
            rule = row["feature_rule"]
            weight = row["weight"]
            for fn in feature_names:
                if fn in rule:
                    lime_dict[fn] = weight
                    break

        fig_cmp = plot_shap_vs_lime(shap_dict, lime_dict, top_k=12)
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Compute both SHAP (Global page) and LIME (above) to see comparison.")
