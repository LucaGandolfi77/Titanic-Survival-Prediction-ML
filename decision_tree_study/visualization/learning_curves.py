"""
Figure: Learning curves (accuracy vs dataset size)
=====================================================
One line per pruning strategy showing how test accuracy scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import apply_style, save_fig, strategy_color, strategy_label


def plot_learning_curves(
    df: pd.DataFrame,
    filename: str = "learning_curves",
) -> Path:
    """Line plot of test accuracy vs subsample_size per strategy."""
    apply_style()
    datasets = df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]
        for strat in sub["strategy"].unique():
            s_df = sub[sub["strategy"] == strat]
            grouped = s_df.groupby("subsample_size")["test_accuracy"].agg(["mean", "std"])
            ax.plot(
                grouped.index,
                grouped["mean"],
                marker="o",
                label=strategy_label(strat),
                color=strategy_color(strat),
                linewidth=2,
            )
            ax.fill_between(
                grouped.index,
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.12,
                color=strategy_color(strat),
            )
        ax.set_title(ds)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=8)

    fig.suptitle("Learning Curves — Accuracy vs Dataset Size", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)
