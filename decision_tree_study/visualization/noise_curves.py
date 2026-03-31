"""
Figure: Noise degradation curves
==================================
Test accuracy vs label-noise rate and vs feature-noise sigma,
one line per pruning strategy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import apply_style, save_fig, strategy_color, strategy_label


def plot_label_noise_curves(
    df: pd.DataFrame,
    filename: str = "noise_label_curves",
) -> Path:
    """Accuracy vs label noise rate per strategy (one subplot per noise mode)."""
    apply_style()
    modes = df["noise_mode"].unique()
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5), squeeze=False)

    for m_idx, mode in enumerate(modes):
        ax = axes[0, m_idx]
        sub = df[df["noise_mode"] == mode]
        for strat in sub["strategy"].unique():
            s_df = sub[sub["strategy"] == strat]
            grouped = s_df.groupby("noise_rate")["test_accuracy"].agg(["mean", "std"])
            ax.plot(
                grouped.index, grouped["mean"],
                marker="o", label=strategy_label(strat),
                color=strategy_color(strat), linewidth=2,
            )
            ax.fill_between(
                grouped.index,
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.12, color=strategy_color(strat),
            )
        ax.set_title(f"Label Noise — {mode}")
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=8)

    fig.suptitle("Label Noise Degradation", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)


def plot_feature_noise_curves(
    df: pd.DataFrame,
    filename: str = "noise_feature_curves",
) -> Path:
    """Accuracy vs feature noise sigma per strategy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in df["strategy"].unique():
        s_df = df[df["strategy"] == strat]
        grouped = s_df.groupby("sigma_factor")["test_accuracy"].agg(["mean", "std"])
        ax.plot(
            grouped.index, grouped["mean"],
            marker="o", label=strategy_label(strat),
            color=strategy_color(strat), linewidth=2,
        )
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.12, color=strategy_color(strat),
        )

    ax.set_xlabel("Feature noise σ-factor")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Feature Noise Degradation")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, filename)
