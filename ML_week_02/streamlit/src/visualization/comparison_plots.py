"""
comparison_plots.py – Side-by-side model comparison charts.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.evaluator import EvalResult


def metric_comparison_bar(
    eval_results: List[EvalResult],
    metric: str = "accuracy",
    title: str | None = None,
) -> go.Figure:
    """Grouped bar chart comparing a single metric across models."""
    data = [
        {"model": er.display_name, metric: er.metrics.get(metric, 0)}
        for er in eval_results
    ]
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="model", y=metric,
        color="model",
        title=title or f"Model Comparison – {metric}",
        template="plotly_white",
        text_auto=".4f",
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def multi_metric_radar(
    eval_results: List[EvalResult],
    metrics: List[str] | None = None,
) -> go.Figure:
    """Radar chart overlaying multiple metrics for all models."""
    if not eval_results:
        return go.Figure()

    if metrics is None:
        # Pick metrics shared by all models (exclude train_time)
        all_keys = set(eval_results[0].metrics.keys())
        for er in eval_results[1:]:
            all_keys &= set(er.metrics.keys())
        metrics = sorted(k for k in all_keys if k != "train_time")

    fig = go.Figure()
    for er in eval_results:
        values = [er.metrics.get(m, 0) for m in metrics]
        values += [values[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=er.display_name,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Comparison – Radar",
        template="plotly_white",
        height=500,
    )
    return fig


def training_time_comparison(eval_results: List[EvalResult]) -> go.Figure:
    """Bar chart of training time."""
    data = [
        {"model": er.display_name, "train_time_sec": er.metrics.get("train_time", 0)}
        for er in eval_results
    ]
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="model", y="train_time_sec",
        color="model",
        title="Training Time Comparison",
        template="plotly_white",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False, height=380, yaxis_title="Seconds")
    return fig


def metric_heatmap(eval_results: List[EvalResult]) -> go.Figure:
    """Heatmap with models as rows and metrics as columns."""
    if not eval_results:
        return go.Figure()

    rows = []
    for er in eval_results:
        rows.append({"model": er.display_name, **er.metrics})
    df = pd.DataFrame(rows).set_index("model")

    fig = px.imshow(
        df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        text_auto=".4f",
        color_continuous_scale="RdYlGn",
        title="Metrics Heatmap",
        template="plotly_white",
        aspect="auto",
    )
    fig.update_layout(height=max(300, 50 * len(eval_results)))
    return fig
