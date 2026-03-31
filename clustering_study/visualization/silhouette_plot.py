"""Silhouette analysis plot — per-cluster silhouette diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score

from visualization._common import _save


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouette Analysis",
    filename: str = "silhouette",
) -> Path:
    n_clusters = len(np.unique(labels))
    sil_avg = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for i in range(n_clusters):
        cluster_sil = np.sort(sample_sil[labels == i])
        size = len(cluster_sil)
        y_upper = y_lower + size
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(sil_avg, color="red", linestyle="--",
               label=f"Mean = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    ax.set_yticks([])
    ax.legend(loc="upper right")
    return _save(fig, filename, subdir="k_selection")
