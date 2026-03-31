"""
Internal Validation Indices
==============================
Metrics that do not require ground-truth labels.
- Silhouette Score (higher is better)
- Calinski-Harabasz Index (higher is better)
- Davies-Bouldin Index (lower is better)
- Dunn Index (higher is better)
- WCSS / Inertia (lower is better)
- BCSS (higher is better)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Within-Cluster Sum of Squares."""
    total = 0.0
    for c in np.unique(labels):
        members = X[labels == c]
        centroid = members.mean(axis=0)
        total += float(np.sum((members - centroid) ** 2))
    return total


def _bcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Between-Cluster Sum of Squares."""
    global_mean = X.mean(axis=0)
    total = 0.0
    for c in np.unique(labels):
        members = X[labels == c]
        centroid = members.mean(axis=0)
        total += len(members) * float(np.sum((centroid - global_mean) ** 2))
    return total


def _dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Dunn index: min inter-cluster dist / max intra-cluster diameter."""
    clusters = np.unique(labels)
    k = len(clusters)
    if k < 2:
        return 0.0

    # intra-cluster diameters (max pairwise distance within cluster)
    diameters = []
    centroids = []
    for c in clusters:
        members = X[labels == c]
        centroids.append(members.mean(axis=0))
        if len(members) < 2:
            diameters.append(0.0)
        else:
            # approximate: max distance from centroid × 2
            dists = np.linalg.norm(members - centroids[-1], axis=1)
            diameters.append(float(2 * np.max(dists)))

    max_diameter = max(diameters)
    if max_diameter == 0:
        return 0.0

    # inter-cluster distances (centroid-to-centroid)
    min_inter = float("inf")
    centroids_arr = np.array(centroids)
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.linalg.norm(centroids_arr[i] - centroids_arr[j]))
            min_inter = min(min_inter, d)

    return min_inter / max_diameter


def compute_internal_indices(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute all internal validation indices."""
    n_unique = len(np.unique(labels))

    if n_unique < 2 or n_unique >= len(X):
        return {
            "silhouette": -1.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float("inf"),
            "dunn": 0.0,
            "wcss": _wcss(X, labels),
            "bcss": _bcss(X, labels),
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "dunn": _dunn_index(X, labels),
        "wcss": _wcss(X, labels),
        "bcss": _bcss(X, labels),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    X, y = make_blobs(n_samples=300, centers=4, random_state=0)
    labels = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(X)
    indices = compute_internal_indices(X, labels)
    for k, v in indices.items():
        print(f"  {k:25s} = {v:.4f}")
