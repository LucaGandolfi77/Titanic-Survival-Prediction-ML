"""ISODATA / Adaptive split-merge evolution over iterations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _save


def plot_split_merge_evolution(
    history: List[Dict],
    title: str = "Split/Merge Evolution",
    filename: str = "split_merge_evolution",
) -> Path:
    iterations = [h["iteration"] for h in history]
    k_vals = [h.get("k", h.get("k_after", 0)) for h in history]
    splits = [h.get("splits", 0) for h in history]
    merges = [h.get("merges", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(iterations, k_vals, "o-", color="#1f77b4", linewidth=2)
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title(title)

    width = 0.35
    x = np.arange(len(iterations))
    ax2.bar(x - width / 2, splits, width, label="Splits", color="#2ca02c")
    ax2.bar(x + width / 2, merges, width, label="Merges", color="#d62728")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(iterations)
    ax2.legend()

    fig.tight_layout()
    return _save(fig, filename, subdir="adaptive")
