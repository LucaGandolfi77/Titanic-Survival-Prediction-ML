"""
shap_plots.py – Interactive SHAP-based visualisation helpers (Plotly).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Global: feature importance bar chart ──────────────────────

def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    title: str = "Global Feature Importance (mean |SHAP|)",
) -> go.Figure:
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)

    idx = np.argsort(mean_abs)[-top_k:]
    fig = go.Figure(
        go.Bar(
            x=mean_abs[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            marker_color="#4A90D9",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        height=max(350, top_k * 28),
        margin=dict(l=20, r=20, t=50, b=30),
        template="plotly_white",
    )
    return fig


# ── Global: bee-swarm / dot plot ──────────────────────────────

def plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_k: int = 15,
    title: str = "SHAP Beeswarm Plot",
) -> go.Figure:
    """Dot-strip (beeswarm-style) plot colored by feature value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    records = []
    for rank, fi in enumerate(top_idx):
        name = X.columns[fi]
        vals = shap_values[:, fi]
        if vals.ndim > 1:
            vals = vals[:, 0]  # binary class → take class-1
        fv = X.iloc[:, fi].values
        for sv, feat_val in zip(vals, fv):
            records.append({"feature": name, "shap_value": sv, "feature_value": feat_val, "rank": rank})

    df = pd.DataFrame(records)
    fig = px.strip(
        df,
        x="shap_value",
        y="feature",
        color="feature_value",
        color_continuous_scale="RdBu_r",
        title=title,
    )
    fig.update_layout(
        height=max(400, top_k * 32),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        coloraxis_colorbar_title="Feature<br>Value",
    )
    return fig


# ── Local: waterfall chart for a single instance ─────────────

def plot_waterfall(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    instance_idx: int = 0,
    top_k: int = 12,
    title: str = "SHAP Waterfall – Instance Explanation",
) -> go.Figure:
    """Waterfall plot showing cumulative SHAP contributions."""
    sv = shap_values[instance_idx]
    if sv.ndim > 1:
        sv = sv[:, 1]

    order = np.argsort(np.abs(sv))[::-1][:top_k]
    order = order[::-1]

    names = [feature_names[i] for i in order]
    values = [sv[i] for i in order]
    cumulative = base_value
    measures = []
    xs = []
    texts = []

    for v in values:
        measures.append("relative")
        xs.append(v)
        cumulative += v
        texts.append(f"{v:+.3f}")

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=names,
            x=xs,
            measure=measures,
            text=texts,
            textposition="outside",
            base=base_value,
            connector_line_color="rgba(0,0,0,0)",
            increasing_marker_color="#D9534F",
            decreasing_marker_color="#4A90D9",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on prediction)",
        height=max(350, top_k * 30),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Local: force-plot style (horizontal stacked) ─────────────

def plot_force_horizontal(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    instance_idx: int = 0,
    top_k: int = 10,
    title: str = "SHAP Force Plot",
) -> go.Figure:
    """Simplified horizontal force plot using stacked bars."""
    sv = shap_values[instance_idx]
    if sv.ndim > 1:
        sv = sv[:, 1]

    order = np.argsort(np.abs(sv))[::-1][:top_k]
    pos = [(feature_names[i], sv[i]) for i in order if sv[i] > 0]
    neg = [(feature_names[i], sv[i]) for i in order if sv[i] <= 0]

    fig = go.Figure()

    # Positive contributions
    if pos:
        fig.add_trace(go.Bar(
            y=["Prediction"],
            x=[sum(v for _, v in pos)],
            name="↑ Increases prediction",
            marker_color="#D9534F",
            orientation="h",
            text=", ".join(f"{n}" for n, _ in pos[:3]),
            textposition="inside",
        ))

    # Negative contributions
    if neg:
        fig.add_trace(go.Bar(
            y=["Prediction"],
            x=[sum(v for _, v in neg)],
            name="↓ Decreases prediction",
            marker_color="#4A90D9",
            orientation="h",
            text=", ".join(f"{n}" for n, _ in neg[:3]),
            textposition="inside",
        ))

    fig.update_layout(
        title=title,
        barmode="relative",
        xaxis_title="SHAP contribution",
        height=200,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    # Add base-value annotation
    fig.add_vline(x=base_value, line_dash="dash", line_color="gray",
                  annotation_text=f"base = {base_value:.3f}")
    return fig


# ── SHAP dependence / interaction scatter ─────────────────────

def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """SHAP dependence scatter for a single feature."""
    fi = list(X.columns).index(feature)
    sv = shap_values[:, fi]
    if sv.ndim > 1:
        sv = sv[:, 1]

    color_vals = X[interaction_feature].values if interaction_feature else None

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=X[feature].values,
        y=sv,
        mode="markers",
        marker=dict(
            size=4,
            color=color_vals,
            colorscale="RdBu_r" if color_vals is not None else None,
            showscale=color_vals is not None,
            colorbar_title=interaction_feature,
            opacity=0.6,
        ),
    ))
    fig.update_layout(
        title=title or f"SHAP Dependence – {feature}",
        xaxis_title=feature,
        yaxis_title="SHAP value",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig
