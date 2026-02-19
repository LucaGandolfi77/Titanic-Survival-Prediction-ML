"""
results_display.py – Streamlit components for showing evaluation results.
"""
from __future__ import annotations

from typing import List

import streamlit as st
import pandas as pd

from src.models.evaluator import EvalResult, comparison_dataframe


def render_metrics_table(eval_results: List[EvalResult]) -> None:
    """Sortable metrics comparison table."""
    if not eval_results:
        st.warning("No results to display.")
        return

    df = comparison_dataframe(eval_results)
    st.subheader("📊 Metrics Comparison")
    st.dataframe(
        df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda").highlight_min(axis=0, color="#f8d7da"),
        use_container_width=True,
    )


def render_metric_cards(eval_result: EvalResult) -> None:
    """Display a single model's metrics as Streamlit metric cards."""
    st.markdown(f"### {eval_result.display_name}")
    cols = st.columns(min(len(eval_result.metrics), 5))
    for i, (name, value) in enumerate(eval_result.metrics.items()):
        with cols[i % len(cols)]:
            st.metric(name.replace("_", " ").title(), f"{value:.4f}")


def render_classification_report(eval_result: EvalResult) -> None:
    """Show the sklearn classification report in a code block."""
    if eval_result.report:
        with st.expander(f"📝 Classification Report – {eval_result.display_name}"):
            st.code(eval_result.report)


def render_best_model_banner(eval_results: List[EvalResult], metric: str = "accuracy") -> None:
    """Highlight the best model in a green banner."""
    if not eval_results:
        return

    best = max(eval_results, key=lambda er: er.metrics.get(metric, 0))
    value = best.metrics.get(metric, 0)
    st.success(f"🏆 **Best Model:** {best.display_name} — {metric} = **{value:.4f}**")
