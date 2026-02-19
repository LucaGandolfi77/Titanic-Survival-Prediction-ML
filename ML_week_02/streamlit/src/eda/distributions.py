"""
distributions.py – Histogram & box-plot helpers (Plotly).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram(
    df: pd.DataFrame,
    column: str,
    nbins: int = 30,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive histogram for a single column."""
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        color=color,
        title=title or f"Distribution of {column}",
        marginal="box",
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05, height=420)
    return fig


def box_plot(
    df: pd.DataFrame,
    column: str,
    group_by: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive box plot."""
    fig = px.box(
        df,
        y=column,
        x=group_by,
        color=group_by,
        title=title or f"Box plot – {column}",
        template="plotly_white",
        points="outliers",
    )
    fig.update_layout(height=420)
    return fig


def all_numeric_histograms(df: pd.DataFrame, ncols: int = 3) -> go.Figure:
    """Grid of histograms for every numeric column."""
    num_cols = df.select_dtypes("number").columns.tolist()
    if not num_cols:
        return go.Figure()

    nrows = -(-len(num_cols) // ncols)  # ceil division
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=num_cols)

    for idx, col in enumerate(num_cols):
        r, c = divmod(idx, ncols)
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        height=280 * nrows,
        template="plotly_white",
        title_text="Numeric Feature Distributions",
    )
    return fig


def violin_plot(
    df: pd.DataFrame,
    column: str,
    group_by: str | None = None,
) -> go.Figure:
    """Violin plot."""
    fig = px.violin(
        df,
        y=column,
        x=group_by,
        color=group_by,
        box=True,
        points="all",
        template="plotly_white",
        title=f"Violin – {column}",
    )
    fig.update_layout(height=420)
    return fig
