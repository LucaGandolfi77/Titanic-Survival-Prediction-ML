"""
Surrogate Quality
=================
Predicted vs actual accuracy scatter and Spearman ρ curve over generations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_surrogate_scatter(
    actual: List[float],
    predicted: List[float],
    save_path: Optional[Path] = None,
    title: str = "Surrogate: Predicted vs Actual Accuracy",
) -> None:
    """Scatter plot of predicted vs actual with diagonal reference."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)

    lo = min(min(actual), min(predicted)) - 0.02
    hi = max(max(actual), max(predicted)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

    from scipy.stats import spearmanr
    rho, _ = spearmanr(actual, predicted)
    r2 = 1 - np.sum((np.array(actual) - np.array(predicted))**2) / \
         np.sum((np.array(actual) - np.mean(actual))**2)

    ax.text(0.05, 0.92, f"ρ = {rho:.3f}\nR² = {r2:.3f}",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Actual Accuracy")
    ax.set_ylabel("Predicted Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_surrogate_rho_curve(
    rho_values: List[float],
    save_path: Optional[Path] = None,
    title: str = "Surrogate Spearman ρ Over Generations",
) -> None:
    """Line plot of Spearman ρ over surrogate retrain events."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(rho_values)), rho_values, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Retrain Event")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0.8, color="green", linestyle=":", alpha=0.5, label="ρ=0.8")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    actual = rng.uniform(0.5, 0.95, 50).tolist()
    predicted = [a + rng.normal(0, 0.03) for a in actual]
    plot_surrogate_scatter(actual, predicted, Path("/tmp/surr_scatter.png"))
    plot_surrogate_rho_curve([0.3, 0.5, 0.65, 0.72, 0.8, 0.83, 0.85],
                             Path("/tmp/surr_rho.png"))
    print("Surrogate quality plots saved.")
