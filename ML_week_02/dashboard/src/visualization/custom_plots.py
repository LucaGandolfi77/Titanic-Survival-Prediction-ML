"""
custom_plots.py – Generic reusable plots (model performance, data overview, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Confusion matrix ──────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Annotated heatmap confusion matrix."""
    labels = labels or [str(i) for i in range(cm.shape[0])]
    text = [[str(int(v)) for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=text,
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        height=400,
        width=450,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── ROC curve ─────────────────────────────────────────────────

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {auc_score:.3f}",
        line=dict(color="#4A90D9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Precision-Recall curve ───────────────────────────────────

def plot_precision_recall(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_score: float,
    title: str = "Precision-Recall Curve",
) -> go.Figure:
    """PR curve with average-precision annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=f"AP = {ap_score:.3f}",
        line=dict(color="#4A90D9", width=2),
        fill="tozeroy",
        fillcolor="rgba(74, 144, 217, 0.1)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Feature distribution ─────────────────────────────────────

def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Histogram with optional class hue."""
    fig = px.histogram(
        df, x=feature, color=hue,
        barmode="overlay", opacity=0.7,
        title=title or f"Distribution – {feature}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Correlation heatmap ──────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Correlation heatmap for numeric columns."""
    corr = df.select_dtypes(include=[np.number]).corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=title,
        height=max(400, len(corr.columns) * 30 + 80),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Model comparison radar chart ─────────────────────────────

def plot_model_radar(
    metrics: Dict[str, float],
    title: str = "Model Performance Overview",
) -> go.Figure:
    """Radar chart for multiple performance metrics."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1]  # close the polygon
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line_color="#4A90D9",
        fillcolor="rgba(74, 144, 217, 0.3)",
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ── Data overview stats ──────────────────────────────────────

def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing Values by Feature",
) -> go.Figure:
    """Horizontal bar chart of missing-value percentages."""
    missing = df.isnull().mean().sort_values(ascending=True)
    missing = missing[missing > 0]

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values!", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=18)
        fig.update_layout(title=title, height=200)
        return fig

    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index.tolist(),
        orientation="h",
        marker_color="#F5A623",
        text=[f"{v:.1%}" for v in missing.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Missing %",
        xaxis_tickformat=".0%",
        height=max(250, len(missing) * 25 + 80),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig
