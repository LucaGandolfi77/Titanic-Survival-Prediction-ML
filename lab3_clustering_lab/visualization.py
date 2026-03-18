"""visualization.py — All plotting functions for the clustering lab."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path("outputs")


def _ensure_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# Part A — 2D Gaussian scatter plots
# -----------------------------------------------------------------------

def plot_gaussian_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    title: str,
    save_name: str,
    misclassified: np.ndarray | None = None,
) -> None:
    """Scatter plot of 2-D data coloured by cluster.

    Args:
        X: (n, 2) feature matrix.
        cluster_labels: cluster id per sample.
        centroids: (k, 2) centroid coordinates.
        title: plot title.
        save_name: filename inside ``outputs/``.
        misclassified: optional boolean mask; highlighted with red ×.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    unique = sorted(set(cluster_labels))
    cmap = plt.cm.tab10

    for ci in unique:
        mask = cluster_labels == ci
        ax.scatter(X[mask, 0], X[mask, 1], c=[cmap(ci)],
                   label=f"Cluster {ci}", alpha=0.6, s=40)

    if misclassified is not None:
        ax.scatter(X[misclassified, 0], X[misclassified, 1],
                   marker="x", c="red", s=100, linewidths=2,
                   label="Misclassified", zorder=5)

    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker="*", c="black", s=250, edgecolors="white",
               linewidths=1.2, zorder=10, label="Centroids")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = OUTPUTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# -----------------------------------------------------------------------
# Part B — Digit centroid images
# -----------------------------------------------------------------------

def plot_digit_centroids(
    centroids: np.ndarray,
    cluster_class_map: dict[int, str],
    save_name: str = "digit_centroids_kmeans.png",
    rows: int = 2,
    cols: int = 5,
) -> None:
    """Display centroid vectors as 13×8 grayscale images.

    Args:
        centroids: (k, 104) centroid matrix.
        cluster_class_map: mapping cluster_id → majority class label.
        save_name: output filename.
        rows: subplot rows.
        cols: subplot columns.
    """
    _ensure_dir()
    k = centroids.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes_flat = axes.flat

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < k:
            img = centroids[i].reshape(13, 8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            label = cluster_class_map.get(i, "?")
            ax.set_title(f"C{i} → '{label}'", fontsize=9)
        ax.axis("off")

    fig.suptitle("K-Means Centroids (13×8 digit images)", fontsize=12)
    fig.tight_layout()

    path = OUTPUTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# -----------------------------------------------------------------------
# Part C — Overfitting curve
# -----------------------------------------------------------------------

def plot_overfitting_curve(
    k_values: list[int],
    train_acc: list[float],
    test_acc: list[float],
    save_name: str = "xmeans_overfitting.png",
) -> None:
    """Train vs test accuracy as a function of number of clusters.

    Args:
        k_values: number of clusters found for each run.
        train_acc: training accuracy per run.
        test_acc: test accuracy per run.
        save_name: output filename.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, train_acc, "o-", label="Train Accuracy")
    ax.plot(k_values, test_acc, "s-", label="Test Accuracy")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("X-Means: Train vs Test Accuracy (Overfitting Analysis)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = OUTPUTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")
