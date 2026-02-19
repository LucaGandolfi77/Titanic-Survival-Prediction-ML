"""
4_fairness_analysis.py – Fairness & Bias analysis page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.fairness.metrics import compute_all_metrics
from src.fairness.bias_detector import detect_bias, bias_summary_text
from src.fairness.mitigation import (
    compute_sample_weights, find_equalised_thresholds, generate_recommendations,
)
from src.utils.session_state import get, put, put_many, has, stage_ready
from src.visualization.fairness_plots import (
    plot_fairness_overview, plot_group_metrics, plot_disparate_impact_gauge,
    plot_selection_rates, plot_bias_heatmap, plot_mitigation_comparison,
)

st.header("⚖️ Fairness & Bias Analysis")

# ── Guard ─────────────────────────────────────────────────────
if not stage_ready("model"):
    st.warning("⚠️ Please load data and a model on the **Overview** page first.")
    st.stop()

model = get("model")
X_test = get("X_test")
y_test = get("y_test")
protected_attrs = get("protected_attrs")
df = get("dataframe")
target_col = get("target_col")

if X_test is None or y_test is None:
    st.warning("⚠️ No test data. Train a model on the Overview page.")
    st.stop()

if not protected_attrs:
    st.info("⚠️ No protected attributes selected. Go to **Overview** to configure them.")
    st.stop()

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

# ══════════════════════════════════════════════════════════════
# Fairness scan
# ══════════════════════════════════════════════════════════════
st.subheader("1️⃣ Bias Scan")

if st.button("🔎 Run Fairness Scan", key="fairness_scan") or has("fairness_results"):

    all_results = []

    for attr in protected_attrs:
        if attr in X_test.columns:
            sensitive = X_test[attr].values
        elif attr in df.columns:
            # Protected attr might have been excluded from features
            sensitive = df.loc[X_test.index, attr].values if attr in df.columns else None
        else:
            st.warning(f"Attribute '{attr}' not found in data.")
            continue

        if sensitive is None:
            continue

        bias_df = detect_bias(
            y_true=y_test.values,
            y_pred=y_pred,
            sensitive=sensitive,
            attribute_name=attr,
        )
        all_results.append(bias_df)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        put("fairness_results", results_df)
        put("bias_scan", results_df)
        st.success("✅ Fairness scan complete!")
    else:
        st.error("No valid protected attributes found.")
        st.stop()

results_df = get("fairness_results")
if results_df is None:
    st.stop()

# ── Display results ───────────────────────────────────────────
st.subheader("2️⃣ Results")

# Summary text
summary_text = bias_summary_text(results_df)
st.markdown(summary_text)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Table", "📊 Charts", "🎯 Disparate Impact", "🛠️ Mitigation"])

# Results table
with tab1:
    # Color-code status
    def _status_style(val):
        if "✅" in str(val):
            return "background-color: #d4edda"
        elif "⚠️" in str(val):
            return "background-color: #fff3cd"
        elif "❌" in str(val):
            return "background-color: #f8d7da"
        return ""

    styled = results_df.style.applymap(_status_style, subset=["status"])
    st.dataframe(styled, use_container_width=True)

# Charts
with tab2:
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Heatmap
        fig_hm = plot_bias_heatmap(results_df)
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_c2:
        # Selection rates per attribute
        for attr in protected_attrs:
            if attr in X_test.columns:
                sensitive = X_test[attr].values
            elif attr in df.columns:
                sensitive = df.loc[X_test.index, attr].values
            else:
                continue
            groups = np.unique(sensitive)
            rates = {}
            for g in groups:
                mask = sensitive == g
                rates[str(g)] = float(y_pred[mask].mean())
            fig_sr = plot_selection_rates(rates, title=f"Selection Rates – {attr}")
            st.plotly_chart(fig_sr, use_container_width=True)

# Disparate impact gauges
with tab3:
    di_rows = results_df[results_df["metric"] == "disparate_impact_ratio"]
    if not di_rows.empty:
        cols = st.columns(min(len(di_rows), 3))
        for i, (_, row) in enumerate(di_rows.iterrows()):
            with cols[i % len(cols)]:
                fig_g = plot_disparate_impact_gauge(
                    row["value"], row["protected_attribute"],
                )
                st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("No disparate impact metric computed.")

# Mitigation
with tab4:
    st.markdown("### Recommendations")
    recs = generate_recommendations(results_df)
    put("recommendations", recs)

    for rec in recs:
        severity = rec["severity"]
        color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
        with st.expander(f"{color} [{severity}] {rec['category']}", expanded=(severity in ("CRITICAL", "HIGH"))):
            st.markdown(f"**Finding:** {rec['finding']}")
            st.markdown(f"**Action:** {rec['recommendation']}")
            if rec.get("regulation"):
                st.caption(f"📜 {rec['regulation']}")

    st.divider()

    st.markdown("### Threshold Optimisation")
    if y_proba is not None and st.button("⚙️ Find Equalised Thresholds", key="eq_thresh"):
        for attr in protected_attrs:
            if attr in X_test.columns:
                sensitive = X_test[attr].values
            elif attr in df.columns:
                sensitive = df.loc[X_test.index, attr].values
            else:
                continue
            thresholds = find_equalised_thresholds(y_test.values, y_proba, sensitive)
            st.json(thresholds)

    st.markdown("### Re-weighting Preview")
    if st.button("📐 Compute Sample Weights", key="sample_weights"):
        for attr in protected_attrs:
            if attr in X_test.columns:
                sensitive = X_test[attr].values
            elif attr in df.columns:
                sensitive = df.loc[X_test.index, attr].values
            else:
                continue
            weights = compute_sample_weights(y_test.values, sensitive)
            st.markdown(f"**{attr}** — weight range: [{weights.min():.3f}, {weights.max():.3f}]")
            st.bar_chart(pd.Series(weights).value_counts().sort_index())

    st.success("✅ Fairness analysis complete — proceed to **Reports** →")
