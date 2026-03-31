"""Exp 7 — 2-D interaction heatmaps (noise×size, imbalance×outliers)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization._common import _save, method_label


def plot_interaction_heatmap(
    df: pd.DataFrame,
    x_col: str = "noise_rate",
    y_col: str = "n_samples",
    score_col: str = "test_accuracy",
    method_col: str = "method",
    title: str = "Interaction: Noise × Dataset Size",
    filename: str = "interaction_heatmap",
) -> Path:
    methods = sorted(df[method_col].unique())
    n = len(methods)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, m in zip(axes_flat, methods):
        sub = df[df[method_col] == m]
        pivot = sub.pivot_table(values=score_col, index=y_col, columns=x_col, aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGnBu",
                    cbar_kws={"label": score_col.replace("_", " ").title()})
        ax.set_title(method_label(m), fontsize=10)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, filename, subdir="exp7")
