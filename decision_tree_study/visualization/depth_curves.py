"""
Figure: Accuracy vs Max Depth curves
======================================
Line plots showing train / val / test accuracy as max_depth increases.
One subplot per dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import apply_style, save_fig


def plot_depth_curves(df: pd.DataFrame, filename: str = "depth_curves") -> Path:
    """Plot accuracy vs max_depth for each dataset."""
    apply_style()
    datasets = df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]
        grouped = sub.groupby("max_depth")[["train_accuracy", "val_accuracy", "test_accuracy"]].agg(
            ["mean", "std"]
        )
        depths = grouped.index.values

        for col, label, ls in [
            ("train_accuracy", "Train", "--"),
            ("val_accuracy", "Val", "-."),
            ("test_accuracy", "Test", "-"),
        ]:
            mean = grouped[(col, "mean")]
            std = grouped[(col, "std")]
            ax.plot(depths, mean, ls, label=label, linewidth=2)
            ax.fill_between(depths, mean - std, mean + std, alpha=0.15)

        ax.set_title(ds)
        ax.set_xlabel("max_depth (-1 = unlimited)")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Accuracy vs Max Depth", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)
