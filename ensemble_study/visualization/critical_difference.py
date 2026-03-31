"""Nemenyi Critical-Difference diagram."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from evaluation.statistical_tests import nemenyi_critical_difference
from visualization._common import _save, method_label


def plot_critical_difference(
    avg_ranks: Dict[str, float],
    n_datasets: int,
    alpha: float = 0.05,
    title: str = "Nemenyi Critical Difference Diagram",
    filename: str = "critical_difference",
) -> Path:
    cd = nemenyi_critical_difference(n_datasets, len(avg_ranks), alpha)
    sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 3 + 0.3 * len(avg_ranks)))
    k = len(sorted_methods)
    y_positions = np.linspace(0, 1, k)

    # Draw axis
    ranks = [r for _, r in sorted_methods]
    lo, hi = min(ranks) - 0.5, max(ranks) + 0.5
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0.5, color="gray", linewidth=0.5)

    # Draw each method
    for i, (m, r) in enumerate(sorted_methods):
        y = y_positions[i]
        ax.plot(r, y, "ko", markersize=6)
        side = "left" if i % 2 == 0 else "right"
        offset = -0.15 if side == "left" else 0.15
        ax.annotate(f"{method_label(m)} ({r:.2f})", (r, y),
                    xytext=(offset, 0), textcoords="offset fontsize",
                    fontsize=9, va="center", ha=side)

    # Draw CD bar
    ax.plot([1, 1 + cd], [1.1, 1.1], "k-", linewidth=2)
    ax.text(1 + cd / 2, 1.13, f"CD = {cd:.2f}", ha="center", fontsize=9)

    # Draw connections for groups not significantly different
    for i, (m1, r1) in enumerate(sorted_methods):
        for j, (m2, r2) in enumerate(sorted_methods):
            if j > i and abs(r2 - r1) < cd:
                y_mid = (y_positions[i] + y_positions[j]) / 2
                ax.plot([r1, r2], [y_mid, y_mid], "-", color="#2ca02c",
                        linewidth=3, alpha=0.4)

    ax.set_xlabel("Average Rank")
    ax.set_title(title)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, filename)
