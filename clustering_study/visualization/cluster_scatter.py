"""2-D cluster assignment scatter plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _save


def plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray | None = None,
    title: str = "Cluster Assignments",
    filename: str = "cluster_scatter",
    subdir: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10",
                         s=15, alpha=0.7, edgecolors="none")
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="red",
                   marker="X", s=200, edgecolors="k", linewidths=1,
                   label="Centroids")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return _save(fig, filename, subdir=subdir)
