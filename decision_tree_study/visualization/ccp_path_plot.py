"""
Figure: CCP Alpha Path
========================
Train/test accuracy vs ccp_alpha, showing the regularisation trajectory
and the knee-point where test accuracy peaks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import apply_style, save_fig


def plot_ccp_path(
    df: pd.DataFrame,
    filename: str = "ccp_alpha_path",
) -> Path:
    """Line plots of train/test accuracy vs ccp_alpha per noise level."""
    apply_style()
    noise_rates = sorted(df["noise_rate"].unique())
    n = len(noise_rates)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)

    for idx, nr in enumerate(noise_rates):
        ax = axes[0, idx]
        sub = df[df["noise_rate"] == nr]
        grouped = sub.groupby("ccp_alpha")[["train_accuracy", "test_accuracy"]].mean()

        ax.plot(grouped.index, grouped["train_accuracy"], "--", label="Train", linewidth=1.5)
        ax.plot(grouped.index, grouped["test_accuracy"], "-", label="Test", linewidth=2)

        # Mark the peak test accuracy
        best_idx = grouped["test_accuracy"].idxmax()
        best_val = grouped.loc[best_idx, "test_accuracy"]
        ax.axvline(best_idx, color="red", linestyle=":", alpha=0.6)
        ax.annotate(
            f"α*={best_idx:.4f}\nacc={best_val:.3f}",
            xy=(best_idx, best_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red"),
        )

        ax.set_title(f"Noise rate = {nr:.0%}")
        ax.set_xlabel("ccp_alpha")
        ax.set_ylabel("Accuracy")
        ax.legend()

    fig.suptitle("Cost-Complexity Pruning Path", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)
