"""
Pipeline graph: visual block diagram of the best discovered pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from ..search_space.pipeline_builder import describe_pipeline
from ..search_space.space_definition import (
    CLASSIFIER_OPTIONS,
    DIM_REDUCTION_OPTIONS,
    FEATURE_SEL_OPTIONS,
    SCALER_OPTIONS,
    categorical_index,
)


def plot_pipeline_graph(
    chromosome: List[float],
    n_features: int,
    title: str = "Best Pipeline Architecture",
    save_path: Path | None = None,
) -> plt.Figure:
    """Draw a block diagram of the pipeline encoded by the chromosome.

    Args:
        chromosome: The chromosome (list of 13 floats).
        n_features: Number of features in the dataset.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="white")

    blocks = []
    scaler_idx = categorical_index(chromosome[0], len(SCALER_OPTIONS))
    scaler = SCALER_OPTIONS[scaler_idx]
    if scaler != "none":
        blocks.append(("Scaler", scaler.replace("_", "\n"), "#AED6F1"))

    fs_idx = categorical_index(chromosome[1], len(FEATURE_SEL_OPTIONS))
    fs = FEATURE_SEL_OPTIONS[fs_idx]
    if fs != "none":
        k = max(1, int(round(chromosome[2] * (n_features - 1) + 1)))
        blocks.append(("Feature Selection", f"{fs}\nk={k}", "#A9DFBF"))

    dr_idx = categorical_index(chromosome[3], len(DIM_REDUCTION_OPTIONS))
    dr = DIM_REDUCTION_OPTIONS[dr_idx]
    if dr != "none":
        blocks.append(("Dim. Reduction", dr.upper(), "#F9E79F"))

    clf_idx = categorical_index(chromosome[4], len(CLASSIFIER_OPTIONS))
    clf = CLASSIFIER_OPTIONS[clf_idx]
    blocks.append(("Classifier", clf.replace("_", "\n"), "#F5B7B1"))

    fig, ax = plt.subplots(figsize=(3 * len(blocks) + 2, 3))
    ax.set_xlim(-0.5, len(blocks) * 3)
    ax.set_ylim(-1, 2.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i, (label, detail, color) in enumerate(blocks):
        x = i * 3
        rect = mpatches.FancyBboxPatch(
            (x, 0), 2.4, 1.8,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 1.2, 1.4, label, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + 1.2, 0.6, detail, ha="center", va="center",
                fontsize=8, style="italic")

        if i < len(blocks) - 1:
            ax.annotate(
                "", xy=(x + 2.6, 0.9), xytext=(x + 2.4, 0.9),
                arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
            )

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    chrom = [0.33, 0.33, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    plot_pipeline_graph(chrom, n_features=30)
    plt.show()
