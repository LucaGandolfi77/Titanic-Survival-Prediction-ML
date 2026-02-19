"""
fairness_plots.py – Fairness-related visualisations (Plotly).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Metric overview radar / bar chart ─────────────────────────

def plot_fairness_overview(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
    title: str = "Fairness Metrics Overview",
) -> go.Figure:
    """Bar chart of fairness metrics with threshold lines."""
    names = list(metrics.keys())
    values = list(metrics.values())

    colors = []
    for n, v in metrics.items():
        t = (thresholds or {}).get(n)
        if t is None:
            colors.append("#4A90D9")
        elif abs(v) <= t:
            colors.append("#5CB85C")
        elif abs(v) <= t * 1.5:
            colors.append("#F5A623")
        else:
            colors.append("#D9534F")

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )

    if thresholds:
        for name, t in thresholds.items():
            if name in names:
                fig.add_hline(y=t, line_dash="dash", line_color="red",
                              annotation_text=f"threshold = {t}")

    fig.update_layout(
        title=title,
        yaxis_title="Value",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=60),
    )
    return fig


# ── Group performance comparison ──────────────────────────────

def plot_group_metrics(
    group_metrics: pd.DataFrame,
    metric_col: str = "accuracy",
    group_col: str = "group",
    title: Optional[str] = None,
) -> go.Figure:
    """Bar chart comparing a metric across groups."""
    fig = px.bar(
        group_metrics,
        x=group_col,
        y=metric_col,
        color=group_col,
        text=metric_col,
        title=title or f"{metric_col} by Group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=380,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Disparate impact gauge ────────────────────────────────────

def plot_disparate_impact_gauge(
    di_ratio: float,
    attribute_name: str = "",
    title: Optional[str] = None,
) -> go.Figure:
    """Gauge chart for disparate impact ratio (0–1.5 range)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=di_ratio,
        delta={"reference": 0.8, "increasing": {"color": "#5CB85C"},
               "decreasing": {"color": "#D9534F"}},
        gauge=dict(
            axis=dict(range=[0, 1.5]),
            bar=dict(color="#4A90D9"),
            steps=[
                dict(range=[0, 0.8], color="#FDECEA"),
                dict(range=[0.8, 1.0], color="#FFF3CD"),
                dict(range=[1.0, 1.5], color="#D4EDDA"),
            ],
            threshold=dict(line=dict(color="red", width=3), thickness=0.8,
                           value=0.8),
        ),
    ))
    fig.update_layout(
        title=title or f"Disparate Impact – {attribute_name}",
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ── Selection rate comparison ─────────────────────────────────

def plot_selection_rates(
    rates: Dict[str, float],
    threshold: float = 0.1,
    title: str = "Positive Prediction Rate by Group",
) -> go.Figure:
    """Bar chart of selection (positive-prediction) rates per group."""
    overall = sum(rates.values()) / len(rates) if rates else 0
    names = list(rates.keys())
    values = list(rates.values())

    colors = ["#D9534F" if abs(v - overall) > threshold else "#5CB85C"
              for v in values]

    fig = go.Figure(
        go.Bar(
            x=names, y=values, marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.add_hline(y=overall, line_dash="dot", line_color="gray",
                  annotation_text=f"Overall: {overall:.1%}")
    fig.update_layout(
        title=title,
        yaxis_title="Selection Rate",
        yaxis_tickformat=".0%",
        height=380,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Bias scan heatmap ─────────────────────────────────────────

def plot_bias_heatmap(
    scan_df: pd.DataFrame,
    title: str = "Bias Scan – Attribute × Metric",
) -> go.Figure:
    """Heatmap showing bias metric values across all attribute-metric combos."""
    pivot = scan_df.pivot_table(
        index="protected_attribute",
        columns="metric",
        values="value",
        aggfunc="first",
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Attribute: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=max(300, len(pivot.index) * 60 + 100),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Before / After mitigation comparison ─────────────────────

def plot_mitigation_comparison(
    before: Dict[str, float],
    after: Dict[str, float],
    title: str = "Fairness – Before vs After Mitigation",
) -> go.Figure:
    """Grouped bar chart comparing metrics pre/post mitigation."""
    metrics = list(set(before.keys()) | set(after.keys()))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=[before.get(m, 0) for m in metrics],
        name="Before",
        marker_color="#D9534F",
    ))
    fig.add_trace(go.Bar(
        x=metrics,
        y=[after.get(m, 0) for m in metrics],
        name="After",
        marker_color="#5CB85C",
    ))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="Value",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=60),
    )
    return fig
