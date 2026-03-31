"""Elbow / WCSS plot for k selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _save


def plot_elbow(
    k_range: list,
    inertias: list,
    best_k: int | None = None,
    title: str = "Elbow Method",
    filename: str = "elbow",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "o-", color="#1f77b4", linewidth=2)
    if best_k is not None:
        idx = k_range.index(best_k) if best_k in k_range else 0
        ax.axvline(best_k, linestyle="--", color="#d62728", alpha=0.7,
                   label=f"Elbow k={best_k}")
        ax.plot(best_k, inertias[idx], "ro", markersize=12, zorder=5)
        ax.legend()
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title(title)
    ax.set_xticks(k_range)
    return _save(fig, filename, subdir="k_selection")
