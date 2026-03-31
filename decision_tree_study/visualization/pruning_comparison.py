"""
Figure: Pruning strategy comparison (grouped bar + box plots)
==============================================================
Bar chart of mean test accuracy per strategy, plus box-plots to show
the distribution across seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.style import (
    apply_style,
    save_fig,
    strategy_color,
    strategy_label,
)


def plot_pruning_comparison(
    df: pd.DataFrame,
    filename: str = "pruning_comparison",
) -> Path:
    """Grouped bar chart + boxplot of test accuracy per strategy."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- bar chart (mean ± std) ---
    ax = axes[0]
    summary = (
        df.groupby("strategy")["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    colors = [strategy_color(s) for s in summary["strategy"]]
    labels = [strategy_label(s) for s in summary["strategy"]]
    ax.bar(labels, summary["mean"], yerr=summary["std"], color=colors, capsize=4)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Mean Test Accuracy by Strategy")
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis="x", rotation=25)

    # --- box plot ---
    ax = axes[1]
    order = summary["strategy"].tolist()
    palette = {strategy_label(s): strategy_color(s) for s in order}
    plot_df = df.copy()
    plot_df["Strategy"] = plot_df["strategy"].map(strategy_label)
    sns.boxplot(
        data=plot_df,
        x="Strategy",
        y="test_accuracy",
        order=[strategy_label(s) for s in order],
        palette=palette,
        ax=ax,
    )
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Distribution of Test Accuracy")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Pruning Strategy Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)
