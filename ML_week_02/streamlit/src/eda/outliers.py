"""
outliers.py – Outlier detection with IQR & Z-score methods.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def detect_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Detect outliers using the IQR method for every numeric column.

    Returns a boolean DataFrame of the same shape (True = outlier).
    """
    num = df.select_dtypes("number")
    Q1 = num.quantile(0.25)
    Q3 = num.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (num < lower) | (num > upper)


def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using Z-score for every numeric column."""
    num = df.select_dtypes("number")
    z = (num - num.mean()) / num.std()
    return z.abs() > threshold


def outlier_summary(df: pd.DataFrame, method: str = "iqr", **kwargs) -> pd.DataFrame:
    """Per-column outlier counts and percentages."""
    if method == "iqr":
        mask = detect_iqr(df, **kwargs)
    else:
        mask = detect_zscore(df, **kwargs)

    summary = pd.DataFrame({
        "outlier_count": mask.sum(),
        "outlier_pct": (mask.sum() / len(df) * 100).round(2),
        "total_rows": len(df),
    })
    return summary.sort_values("outlier_count", ascending=False)


def outlier_box_plot(df: pd.DataFrame, columns: list[str] | None = None) -> go.Figure:
    """Box plots highlighting outliers for selected columns."""
    cols = columns or df.select_dtypes("number").columns.tolist()[:10]
    melted = df[cols].melt(var_name="feature", value_name="value")
    fig = px.box(
        melted,
        x="feature",
        y="value",
        color="feature",
        title="Outlier Detection – Box Plots",
        template="plotly_white",
        points="outliers",
    )
    fig.update_layout(height=480, showlegend=False)
    return fig
