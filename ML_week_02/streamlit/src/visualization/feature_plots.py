"""
feature_plots.py – Feature importance & distribution visualizations.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def feature_importance_bar(
    imp_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    col = "importance" if "importance" in imp_df.columns else "importance_mean"
    df = imp_df.sort_values(col, ascending=True).tail(top_n)
    fig = px.bar(
        df, x=col, y="feature",
        orientation="h",
        title=title,
        template="plotly_white",
        color=col,
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=max(350, 25 * len(df)),
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return fig


def feature_distribution_by_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> go.Figure:
    """Overlaid histograms of a feature split by target class."""
    fig = px.histogram(
        df, x=feature, color=target,
        barmode="overlay",
        opacity=0.7,
        title=f"{feature} by {target}",
        template="plotly_white",
        marginal="box",
    )
    fig.update_layout(height=420)
    return fig


def pairwise_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
) -> go.Figure:
    """Scatter-plot of two features."""
    fig = px.scatter(
        df, x=x, y=y, color=color,
        title=f"{x} vs {y}",
        template="plotly_white",
        opacity=0.6,
    )
    fig.update_layout(height=420)
    return fig
