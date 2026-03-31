"""Stability heatmap — ARI across seeds or across k values."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization._common import _save


def plot_stability_heatmap(
    df: pd.DataFrame,
    x_col: str = "method",
    y_col: str = "dataset",
    val_col: str = "mean_pairwise_ari",
    title: str = "Initialisation Stability (Mean Pairwise ARI)",
    filename: str = "stability_heatmap",
) -> Path:
    pivot = df.pivot_table(values=val_col, index=y_col, columns=x_col,
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, filename, subdir="exp3")
