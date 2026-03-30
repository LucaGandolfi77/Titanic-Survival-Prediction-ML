"""
Genome heatmap: gene value frequency across generations.

Shows how the population converges on specific gene values over the
course of evolution, revealing which pipeline components and
hyperparameters the EA prefers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


GENE_NAMES = [
    "Scaler", "FeatureSel", "K_ratio", "DimRed", "Classifier",
    "HP_0", "HP_1", "HP_2", "HP_3", "HP_4", "HP_5", "HP_6", "HP_7",
]


def plot_genome_heatmap(
    population_snapshots: Dict[int, List[List[float]]],
    title: str = "Gene Value Frequencies Across Generations",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a heatmap of mean gene values per generation.

    Args:
        population_snapshots: {generation: [[gene_values], ...]}
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens = sorted(population_snapshots.keys())
    n_genes = len(GENE_NAMES)

    # Sample a subset of generations for readability
    if len(gens) > 20:
        step = max(1, len(gens) // 20)
        gens = gens[::step]

    matrix = np.zeros((len(gens), n_genes))
    for i, gen in enumerate(gens):
        pop = np.array(population_snapshots[gen])
        matrix[i, :] = pop[:, :n_genes].mean(axis=0) if pop.shape[1] >= n_genes else 0.0

    fig, ax = plt.subplots(figsize=(14, max(6, len(gens) * 0.3)))
    sns.heatmap(
        matrix,
        xticklabels=GENE_NAMES,
        yticklabels=[f"Gen {g}" for g in gens],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Gene", fontsize=12)
    ax.set_ylabel("Generation", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    snapshots = {}
    for g in range(0, 30, 5):
        pop = rng.uniform(0, 1, size=(50, 13)).tolist()
        snapshots[g] = pop
    plot_genome_heatmap(snapshots, title="Test Genome Heatmap")
    plt.show()
