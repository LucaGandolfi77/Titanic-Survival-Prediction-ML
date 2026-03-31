"""Grouped bar chart comparing validation metrics across algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_validation_comparison(
    df: pd.DataFrame,
    metrics: Sequence[str],
    dataset_name: str = "",
    filename: str = "validation_comparison",
) -> Path:
    methods = sorted(df["method"].unique())
    n_metrics = len(metrics)
    n_methods = len(methods)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_metrics), 6))
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        means = [sub[met].mean() for met in metrics]
        stds = [sub[met].std() for met in metrics]
        colour = METHOD_COLORS.get(m, None)
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=method_label(m),
            color=colour,
            capsize=3,
        )

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([m.replace("int_", "").replace("ext_", "") for m in metrics],
                       rotation=30, ha="right")
    title = "Validation Metric Comparison"
    if dataset_name:
        title += f" — {dataset_name}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, filename, subdir="comparison")
