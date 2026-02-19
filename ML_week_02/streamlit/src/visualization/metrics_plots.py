"""
metrics_plots.py – Confusion matrix, ROC curve, precision-recall curve.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.models.evaluator import EvalResult


def confusion_matrix_plot(
    eval_result: EvalResult,
    labels: list[str] | None = None,
) -> go.Figure:
    """Interactive annotated confusion matrix heatmap."""
    cm = eval_result.confusion
    if cm is None:
        return go.Figure()

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    # Normalize for color, but show raw counts as text
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Reds",
        showscale=True,
        colorbar=dict(title="Ratio"),
    ))
    fig.update_layout(
        title=f"Confusion Matrix – {eval_result.display_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def roc_curve_plot(
    eval_results: List[EvalResult],
    y_test: np.ndarray | pd.Series,
) -> go.Figure:
    """Overlay ROC curves for multiple binary classifiers."""
    fig = go.Figure()

    for er in eval_results:
        if er.y_prob is None:
            continue
        prob = er.y_prob[:, 1] if er.y_prob.ndim == 2 else er.y_prob
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{er.display_name} (AUC={roc_auc:.3f})",
        ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="grey"),
        showlegend=False,
    ))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=480,
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def precision_recall_plot(
    eval_results: List[EvalResult],
    y_test: np.ndarray | pd.Series,
) -> go.Figure:
    """Overlay precision-recall curves."""
    fig = go.Figure()

    for er in eval_results:
        if er.y_prob is None:
            continue
        prob = er.y_prob[:, 1] if er.y_prob.ndim == 2 else er.y_prob
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{er.display_name} (AP={ap:.3f})",
        ))

    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=480,
    )
    return fig


def residual_plot(
    eval_result: EvalResult,
    y_test: np.ndarray | pd.Series,
) -> go.Figure:
    """Residual plot for regression models."""
    y_pred = eval_result.y_pred
    residuals = np.array(y_test) - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(opacity=0.5, color="#FF4B4B"),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=f"Residuals – {eval_result.display_name}",
        xaxis_title="Predicted",
        yaxis_title="Residual",
        template="plotly_white",
        height=420,
    )
    return fig


def actual_vs_predicted(
    eval_result: EvalResult,
    y_test: np.ndarray | pd.Series,
) -> go.Figure:
    """Actual vs Predicted scatter for regression."""
    y_true = np.array(y_test)
    y_pred = eval_result.y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(opacity=0.5, color="#FF4B4B"),
        name="Predictions",
    ))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(dash="dash", color="grey"),
        name="Perfect",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted – {eval_result.display_name}",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_white",
        height=420,
    )
    return fig
