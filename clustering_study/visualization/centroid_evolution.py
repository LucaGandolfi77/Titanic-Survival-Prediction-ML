"""Centroid movement during K-Means iterations (2-D view)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from visualization._common import PALETTE, _save


def plot_centroid_evolution(
    X: NDArray[np.floating],
    centroid_history: List[NDArray[np.floating]],
    labels_final: NDArray[np.integer] | None = None,
    title: str = "Centroid Evolution",
    filename: str = "centroid_evolution",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels_final is not None:
        unique = np.unique(labels_final)
        cmap = plt.cm.get_cmap("tab10", len(unique))
        for idx, lab in enumerate(unique):
            mask = labels_final == lab
            ax.scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.25, color=cmap(idx))

    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.2, color="grey")

    n_steps = len(centroid_history)
    for step_idx, centroids in enumerate(centroid_history):
        alpha = 0.3 + 0.7 * step_idx / max(n_steps - 1, 1)
        size = 40 + 100 * step_idx / max(n_steps - 1, 1)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=size,
            edgecolors="black",
            linewidths=0.5,
            alpha=alpha,
            zorder=5,
            color=PALETTE[step_idx % len(PALETTE)],
        )

    for j in range(centroid_history[0].shape[0]):
        path = np.array([ch[j] for ch in centroid_history if j < ch.shape[0]])
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], "k--", linewidth=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    fig.tight_layout()
    return _save(fig, filename, subdir="centroids")
