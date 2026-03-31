"""PCA 2-D / 3-D projection of clustering results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from data.preprocessor import reduce_dimensions
from visualization._common import _save


def plot_pca_projection(
    X: NDArray[np.floating],
    labels: NDArray[np.integer],
    n_components: int = 2,
    title: str = "PCA Projection",
    filename: str = "pca_projection",
) -> Path:
    X_red, pca = reduce_dimensions(X, n_components=n_components)
    explained = pca.explained_variance_ratio_

    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique))

    if n_components == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for idx, lab in enumerate(unique):
            mask = labels == lab
            ax.scatter(
                X_red[mask, 0], X_red[mask, 1], X_red[mask, 2],
                s=15, alpha=0.6, color=cmap(idx), label=f"C{lab}",
            )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_zlabel(f"PC3 ({explained[2]:.1%})")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, lab in enumerate(unique):
            mask = labels == lab
            ax.scatter(
                X_red[mask, 0], X_red[mask, 1],
                s=15, alpha=0.6, color=cmap(idx), label=f"C{lab}",
            )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")

    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, filename, subdir="pca")
