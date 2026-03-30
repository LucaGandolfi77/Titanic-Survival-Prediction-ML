"""
Genome Heatmap
==============
Gene-value frequency heatmap across generations showing convergence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_genome_heatmap(
    generation_genomes: Dict[int, List[List[float]]],
    gene_names: List[str] | None = None,
    save_path: Optional[Path] = None,
    title: str = "Gene Value Heatmap Across Generations",
) -> None:
    """Heatmap of mean gene values per generation.

    Args:
        generation_genomes: {generation_idx: list_of_genomes}
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    gens_sorted = sorted(generation_genomes.keys())
    if not gens_sorted:
        return

    n_genes = len(generation_genomes[gens_sorted[0]][0])
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    matrix = np.zeros((len(gens_sorted), n_genes))
    for row, gen in enumerate(gens_sorted):
        genomes = np.array(generation_genomes[gen])
        matrix[row] = genomes.mean(axis=0)

    # Normalize each column to [0, 1] for comparable visualization
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    matrix_norm = (matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=(max(12, n_genes * 0.8), max(6, len(gens_sorted) * 0.3)))
    sns.heatmap(
        matrix_norm, ax=ax, cmap="YlOrRd",
        xticklabels=gene_names[:n_genes],
        yticklabels=[str(g) for g in gens_sorted],
        cbar_kws={"label": "Normalized mean value"},
    )
    ax.set_xlabel("Gene")
    ax.set_ylabel("Generation")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    gen_genomes = {g: [rng.uniform(0, 1, 14).tolist() for _ in range(50)]
                   for g in range(20)}
    plot_genome_heatmap(gen_genomes, save_path=Path("/tmp/genome_heatmap.png"))
    print("Genome heatmap saved.")
