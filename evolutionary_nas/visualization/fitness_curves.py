"""
Fitness Curves
==============
Plot best / mean / std fitness over generations with shaded std band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_fitness_evolution(
    history: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Fitness Evolution",
) -> None:
    """Plot best, mean fitness with ±1 std shaded band over generations."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    std = [h["std_fitness"] for h in history]

    mean_arr = np.array(mean)
    std_arr = np.array(std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, label="Best", linewidth=2, color="C0")
    ax.plot(gens, mean, label="Mean", linewidth=1.5, color="C1", linestyle="--")
    ax.fill_between(gens, mean_arr - std_arr, mean_arr + std_arr,
                     alpha=0.2, color="C1", label="±1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    history = [
        {"generation": i, "best_fitness": 0.5 + 0.01 * i,
         "mean_fitness": 0.45 + 0.008 * i, "std_fitness": 0.05}
        for i in range(40)
    ]
    plot_fitness_evolution(history, Path("/tmp/test_fitness.png"))
    print("Saved test plot.")
