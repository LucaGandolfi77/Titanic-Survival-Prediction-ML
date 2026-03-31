"""
Figure: Interaction heatmaps
==============================
2-D heatmaps for noise × depth and noise × dataset-size interactions.
Colour encodes mean test accuracy (or overfitting gap).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.style import apply_style, save_fig


def plot_interaction_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str = "test_accuracy",
    title: str = "Interaction Heatmap",
    filename: str = "interaction_heatmap",
) -> Path:
    """Pivot and plot a heatmap of value_col over (x_col, y_col)."""
    apply_style()
    pivot = df.groupby([y_col, x_col])[value_col].mean().reset_index()
    matrix = pivot.pivot(index=y_col, columns=x_col, values=value_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": value_col},
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return save_fig(fig, filename)


def plot_noise_depth_heatmap(
    df: pd.DataFrame,
    filename: str = "heatmap_noise_depth",
) -> Path:
    return plot_interaction_heatmap(
        df,
        x_col="max_depth",
        y_col="noise_rate",
        value_col="test_accuracy",
        title="Noise Rate × Max Depth → Test Accuracy",
        filename=filename,
    )


def plot_noise_size_heatmap(
    df: pd.DataFrame,
    filename: str = "heatmap_noise_size",
) -> Path:
    return plot_interaction_heatmap(
        df,
        x_col="subsample_size",
        y_col="noise_rate",
        value_col="test_accuracy",
        title="Noise Rate × Dataset Size → Test Accuracy",
        filename=filename,
    )
