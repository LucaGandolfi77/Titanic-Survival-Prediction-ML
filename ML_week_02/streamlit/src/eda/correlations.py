"""
correlations.py – Correlation matrices & heatmaps.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix for numeric features."""
    num = df.select_dtypes("number")
    return num.corr(method=method).round(4)


def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str | None = None,
) -> go.Figure:
    """Interactive heatmap of the correlation matrix."""
    corr = correlation_matrix(df, method=method)

    # Mask upper triangle for cleaner look
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig = px.imshow(
        corr_masked,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title or f"Correlation Heatmap ({method.title()})",
        template="plotly_white",
        aspect="auto",
    )
    fig.update_layout(height=520)
    return fig


def top_correlations(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N most correlated feature pairs (absolute value)."""
    corr = correlation_matrix(df, method=method)

    if target_col and target_col in corr.columns:
        # Correlation with target
        target_corr = (
            corr[target_col]
            .drop(target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        target_corr.columns = ["feature", "abs_correlation"]
        return target_corr

    # All pairs
    pairs = (
        corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs_corr"] = pairs["correlation"].abs()
    return pairs.nlargest(top_n, "abs_corr").reset_index(drop=True)


def scatter_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    color: str | None = None,
    max_cols: int = 6,
) -> go.Figure:
    """Pair-plot / scatter matrix for selected numeric columns."""
    num_cols = columns or df.select_dtypes("number").columns.tolist()[:max_cols]
    fig = px.scatter_matrix(
        df,
        dimensions=num_cols,
        color=color,
        template="plotly_white",
        title="Scatter Matrix",
        opacity=0.6,
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=180 * len(num_cols))
    return fig
