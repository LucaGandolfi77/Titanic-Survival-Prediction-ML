"""
📈 Results – Detailed comparison & visualisation of trained models.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np

from src.utils.session_state import init_session_state, has_results
from src.ui.sidebar import render_sidebar
from src.ui.results_display import (
    render_metrics_table,
    render_metric_cards,
    render_classification_report,
    render_best_model_banner,
)
from src.visualization.metrics_plots import (
    confusion_matrix_plot,
    roc_curve_plot,
    precision_recall_plot,
    residual_plot,
    actual_vs_predicted,
)
from src.visualization.comparison_plots import (
    metric_comparison_bar,
    multi_metric_radar,
    training_time_comparison,
    metric_heatmap,
)
from src.models.explainer import get_feature_importance, plot_feature_importance, shap_summary
from src.visualization.decision_boundary import decision_boundary_2d
from src.utils.export import model_download_button, predictions_download_button

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Results", page_icon="📈", layout="wide")
init_session_state()
render_sidebar()

st.title("📈 Results Dashboard")

if not has_results():
    st.info("⬅️ Train models on the **🤖 Model Training** page first.")
    st.stop()

eval_results = st.session_state.eval_results
y_test = st.session_state.y_test
X_test = st.session_state.X_test
X_train = st.session_state.X_train
y_train = st.session_state.y_train
task = st.session_state.task_type

# ═══════════════════════════════════════════════════════════════
# Best model banner
# ═══════════════════════════════════════════════════════════════
primary_metric = "accuracy" if task == "classification" else "r2"
render_best_model_banner(eval_results, metric=primary_metric)

# ═══════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════
tab_comp, tab_detail, tab_explain, tab_dl = st.tabs(
    ["Comparison", "Detailed View", "Explainability", "Download"]
)

# ── Comparison ────────────────────────────────────────────────
with tab_comp:
    render_metrics_table(eval_results)

    st.plotly_chart(metric_heatmap(eval_results), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            metric_comparison_bar(eval_results, metric=primary_metric),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(training_time_comparison(eval_results), use_container_width=True)

    st.plotly_chart(multi_metric_radar(eval_results), use_container_width=True)

    if task == "classification":
        st.subheader("ROC Curves")
        st.plotly_chart(roc_curve_plot(eval_results, y_test), use_container_width=True)
        st.subheader("Precision-Recall Curves")
        st.plotly_chart(precision_recall_plot(eval_results, y_test), use_container_width=True)

# ── Detailed View ─────────────────────────────────────────────
with tab_detail:
    model_choice = st.selectbox(
        "Select model",
        [er.display_name for er in eval_results],
        key="detail_model",
    )
    er = next(er for er in eval_results if er.display_name == model_choice)

    render_metric_cards(er)

    if task == "classification":
        render_classification_report(er)
        st.plotly_chart(confusion_matrix_plot(er), use_container_width=True)

        # Decision boundary
        if X_train is not None:
            with st.expander("🗺️ Decision Boundary (PCA 2-D)"):
                tr = next(tr for tr in st.session_state.train_results if tr.model_key == er.model_key)
                st.plotly_chart(
                    decision_boundary_2d(tr.estimator, X_train, y_train, X_test, y_test),
                    use_container_width=True,
                )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(residual_plot(er, y_test), use_container_width=True)
        with col2:
            st.plotly_chart(actual_vs_predicted(er, y_test), use_container_width=True)

# ── Explainability ────────────────────────────────────────────
with tab_explain:
    model_choice_exp = st.selectbox(
        "Select model",
        [er.display_name for er in eval_results],
        key="explain_model",
    )
    er_exp = next(er for er in eval_results if er.display_name == model_choice_exp)
    tr_exp = next(tr for tr in st.session_state.train_results if tr.model_key == er_exp.model_key)
    feature_names = X_test.columns.tolist() if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_test.shape[1])]

    st.subheader("Feature Importance")
    imp_df = get_feature_importance(tr_exp.estimator, feature_names, X_test, y_test)
    if not imp_df.empty:
        st.plotly_chart(plot_feature_importance(imp_df), use_container_width=True)
    else:
        st.caption("Feature importance not available for this model.")

    st.subheader("SHAP Analysis")
    with st.spinner("Computing SHAP values …"):
        shap_fig = shap_summary(tr_exp.estimator, X_test)
    if shap_fig:
        st.plotly_chart(shap_fig, use_container_width=True)
    else:
        st.caption("SHAP analysis not available for this model type.")

# ── Download ──────────────────────────────────────────────────
with tab_dl:
    st.subheader("⬇️ Download Trained Models & Predictions")
    for i, er in enumerate(eval_results):
        st.markdown(f"**{er.display_name}**")
        tr = next(tr for tr in st.session_state.train_results if tr.model_key == er.model_key)
        col1, col2 = st.columns(2)
        with col1:
            model_download_button(tr.estimator, er.display_name, key=f"dl_m_{i}")
        with col2:
            predictions_download_button(er.y_pred, er.display_name, key=f"dl_p_{i}")
        st.markdown("---")
