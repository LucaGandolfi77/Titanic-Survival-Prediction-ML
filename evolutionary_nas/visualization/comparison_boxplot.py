"""
Comparison Boxplot
==================
Boxplots comparing accuracy/F1 distributions across methods.
Also: diversity curve and early stopping savings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_comparison_boxplot(
    results: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Accuracy Distribution Comparison",
    ylabel: str = "Accuracy",
) -> None:
    """Boxplot comparing distributions across methods."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for method, values in results.items():
        data.extend(values)
        labels.extend([method] * len(values))

    import pandas as pd
    df = pd.DataFrame({"Method": labels, ylabel: data})
    sns.boxplot(data=df, x="Method", y=ylabel, ax=ax, palette="Set2")
    sns.stripplot(data=df, x="Method", y=ylabel, ax=ax, color="black",
                  alpha=0.4, size=4, jitter=True)
    ax.set_title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_curve(
    history: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Population Diversity Over Generations",
) -> None:
    """Plot diversity index over generations."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens = [h["generation"] for h in history]
    div = [h["diversity_index"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, div, "o-", linewidth=2, markersize=4, color="C2")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity Index")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_early_stop_savings(
    epochs_saved_per_gen: List[int],
    save_path: Optional[Path] = None,
    title: str = "Epochs Saved by Predictive Early Stopping",
) -> None:
    """Bar chart of epochs saved per generation."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(epochs_saved_per_gen)), epochs_saved_per_gen,
           color="C3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Epochs Saved")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = {
        "NAS-GA": np.random.uniform(0.88, 0.95, 10).tolist(),
        "NSGA-II": np.random.uniform(0.86, 0.94, 10).tolist(),
        "Random": np.random.uniform(0.80, 0.90, 10).tolist(),
    }
    plot_comparison_boxplot(results, Path("/tmp/boxplot.png"))
    print("Comparison boxplot saved.")
