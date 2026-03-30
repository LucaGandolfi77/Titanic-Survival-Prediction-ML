"""
Comparison boxplots and additional visualization utilities.

Includes: F1 distribution boxplots per method/dataset, diversity curves,
and convergence comparison plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_comparison_boxplot(
    results: Dict[str, Dict[str, List[float]]],
    title: str = "F1 Score Distribution by Method",
    save_path: Path | None = None,
) -> plt.Figure:
    """Boxplot of F1 distributions across methods for a single dataset.

    Args:
        results: {method_name: {"f1_scores": [float, ...]}}.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    data = []
    labels = []
    for method, scores_dict in results.items():
        f1s = scores_dict.get("f1_scores", scores_dict.get("f1", []))
        if isinstance(f1s, (list, np.ndarray)):
            data.extend(f1s)
            labels.extend([method] * len(f1s))
        else:
            data.append(f1s)
            labels.append(method)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=labels, y=data, ax=ax, palette="colorblind")
    sns.stripplot(x=labels, y=data, ax=ax, color="black", alpha=0.4, size=4, jitter=True)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("F1 Score (macro)", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diversity_curve(
    history: List[Dict[str, Any]],
    title: str = "Population Diversity",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot population diversity (unique genome ratio) per generation.

    Args:
        history: List of per-generation stat dicts.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens = [h["generation"] for h in history]
    diversity = [h["diversity"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, diversity, linewidth=2, marker="o", markersize=3, color="teal")
    ax.fill_between(gens, 0, diversity, alpha=0.15, color="teal")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Diversity (unique ratio)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence_comparison(
    histories: Dict[str, List[Dict[str, Any]]],
    title: str = "Convergence Comparison",
    save_path: Path | None = None,
) -> plt.Figure:
    """Compare convergence of multiple evolutionary strategies.

    Args:
        histories: {method_name: list_of_gen_stats}.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("colorblind", n_colors=len(histories))

    for (method, history), color in zip(histories.items(), colors):
        gens = [h["generation"] for h in history]
        best = [h["best_fitness"] for h in history]
        ax.plot(gens, best, label=method, linewidth=2, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best F1 Score (macro)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    sample = {
        "GA": {"f1_scores": np.random.normal(0.92, 0.02, 10).tolist()},
        "NSGA-II": {"f1_scores": np.random.normal(0.90, 0.03, 10).tolist()},
        "Random": {"f1_scores": np.random.normal(0.87, 0.04, 10).tolist()},
    }
    plot_comparison_boxplot(sample)
    plt.show()
