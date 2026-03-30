"""
Fitness evolution curves: best, mean, and std F1 per generation.

Produces a line chart with a shaded standard deviation band, following
a consistent seaborn theme for thesis-quality figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_fitness_evolution(
    history: List[Dict[str, Any]],
    title: str = "Fitness Evolution",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot best/mean fitness with std band over generations.

    Args:
        history: List of per-generation stat dicts (from EvolutionLogger).
        title: Plot title.
        save_path: If provided, save the figure as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    std = [h["std_fitness"] for h in history]

    mean_arr = np.array(mean)
    std_arr = np.array(std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, label="Best F1", linewidth=2, marker="o", markersize=3)
    ax.plot(gens, mean, label="Mean F1", linewidth=1.5, linestyle="--")
    ax.fill_between(
        gens,
        mean_arr - std_arr,
        mean_arr + std_arr,
        alpha=0.2,
        label="± 1 Std Dev",
    )
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("F1 Score (macro)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    fake_history = [
        {"generation": i, "best_fitness": 0.5 + 0.4 * (1 - np.exp(-i / 10)),
         "mean_fitness": 0.4 + 0.3 * (1 - np.exp(-i / 10)),
         "std_fitness": 0.05 * np.exp(-i / 20)}
        for i in range(30)
    ]
    plot_fitness_evolution(fake_history, title="Test Fitness Evolution")
    plt.show()
