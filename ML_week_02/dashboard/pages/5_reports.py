"""
5_reports.py – Generate and download executive & technical reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st

from src.reporting.summary_generator import (
    generate_executive_summary, generate_technical_report, to_plain_text,
)
from src.reporting.pdf_exporter import render_executive_html, render_technical_html
from src.utils.session_state import get, put, has, stage_ready

st.header("📊 Report Generation")

# ── Guard ─────────────────────────────────────────────────────
if not stage_ready("model"):
    st.warning("⚠️ Please complete data loading and model training first (Overview page).")
    st.stop()

model_info = get("model_info") or {}
performance = get("performance") or {}
fairness_results = get("fairness_results")
recommendations = get("recommendations")
shap_vals = get("shap_values")
feature_names = get("feature_cols")

# ── Derive top features from SHAP ─────────────────────────────
import numpy as np
top_features = None
if shap_vals is not None and feature_names:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)
    order = np.argsort(mean_abs)[::-1]
    top_features = [feature_names[i] for i in order[:10]]

# ══════════════════════════════════════════════════════════════
# Executive summary
# ══════════════════════════════════════════════════════════════
st.subheader("1️⃣ Executive Summary")

if st.button("📝 Generate Executive Summary", key="gen_exec") or has("executive_summary"):
    summary = generate_executive_summary(
        model_info=model_info,
        performance=performance,
        fairness_results=fairness_results,
        top_features=top_features,
    )
    put("executive_summary", summary)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", summary["model_type"])
    with col2:
        st.metric(summary["primary_metric_name"].title(), f"{summary['primary_metric_value']:.4f}")
    with col3:
        st.metric("Features", summary["n_features"])
    with col4:
        if summary["fairness"]:
            verdict = summary["fairness"]["verdict"]
            color = {"PASS": "🟢", "WARNING": "🟡", "FAIL": "🔴"}.get(verdict, "⚪")
            st.metric("Fairness", f"{color} {verdict}")
        else:
            st.metric("Fairness", "N/A")

    # Top features
    if summary["top_features"]:
        st.markdown("**Top Influential Features:** " + ", ".join(f"`{f}`" for f in summary["top_features"]))

    # Fairness details
    if summary["fairness"]:
        f = summary["fairness"]
        st.markdown(f"**Fairness Checks:** ✅ {f['pass']} pass · ⚠️ {f['warn']} warn · ❌ {f['fail']} fail")

    # HTML preview + download
    html = render_executive_html(summary)
    with st.expander("Preview HTML"):
        st.components.v1.html(html, height=600, scrolling=True)

    st.download_button(
        "⬇️ Download Executive Summary (HTML)",
        data=html,
        file_name="executive_summary.html",
        mime="text/html",
    )

# ══════════════════════════════════════════════════════════════
# Technical report
# ══════════════════════════════════════════════════════════════
st.subheader("2️⃣ Technical Report")

if st.button("📋 Generate Technical Report", key="gen_tech") or has("technical_report"):
    # SHAP global
    shap_global = None
    if shap_vals is not None and feature_names:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=1)
        shap_global = dict(zip(feature_names, mean_abs.tolist()))

    # Dataset stats
    df = get("dataframe")
    dataset_stats = {}
    if df is not None:
        dataset_stats = {
            "Rows": len(df),
            "Columns": len(df.columns),
            "Target": get("target_col"),
            "Missing %": f"{df.isnull().mean().mean():.1%}",
        }

    report = generate_technical_report(
        model_info=model_info,
        performance=performance,
        shap_global=shap_global,
        fairness_results=fairness_results,
        recommendations=recommendations,
        dataset_stats=dataset_stats,
    )
    put("technical_report", report)

    # HTML preview + download
    html = render_technical_html(report)
    with st.expander("Preview HTML"):
        st.components.v1.html(html, height=800, scrolling=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "⬇️ Download Technical Report (HTML)",
            data=html,
            file_name="technical_report.html",
            mime="text/html",
        )
    with col_d2:
        plain = to_plain_text(report)
        st.download_button(
            "⬇️ Download Plain Text Report",
            data=plain,
            file_name="technical_report.txt",
            mime="text/plain",
        )

# ══════════════════════════════════════════════════════════════
# Quick export all
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("3️⃣ Export All Artifacts")

output_dir = Path(__file__).resolve().parents[1] / "outputs" / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

if st.button("💾 Save All to Disk", key="save_all"):
    saved = []
    if has("executive_summary"):
        html = render_executive_html(get("executive_summary"))
        path = output_dir / "executive_summary.html"
        path.write_text(html, encoding="utf-8")
        saved.append(str(path))
    if has("technical_report"):
        html = render_technical_html(get("technical_report"))
        path = output_dir / "technical_report.html"
        path.write_text(html, encoding="utf-8")
        saved.append(str(path))
        plain = to_plain_text(get("technical_report"))
        path = output_dir / "technical_report.txt"
        path.write_text(plain, encoding="utf-8")
        saved.append(str(path))
    if saved:
        st.success(f"✅ Saved {len(saved)} files to `outputs/reports/`")
        for s in saved:
            st.caption(s)
    else:
        st.warning("Generate reports first!")
