"""
K-Means Wrapper
================
Standard K-Means with diagnostics: per-iteration inertia, centroid
history, and convergence tracking.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from config import CFG


def build_kmeans(
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    **kwargs,
) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )


def fit_kmeans_with_history(
    X: np.ndarray,
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    max_iter: int = 100,
) -> Dict:
    """Fit K-Means tracking centroid positions at each iteration."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # k-means++ initialisation
    km_init = KMeans(n_clusters=n_clusters, init="k-means++",
                     n_init=1, max_iter=1, random_state=random_state)
    km_init.fit(X)
    centroids = km_init.cluster_centers_.copy()

    history: List[np.ndarray] = [centroids.copy()]
    inertia_history: List[float] = []

    for it in range(max_iter):
        # assignment
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.sum(np.min(dists, axis=1) ** 2))
        inertia_history.append(inertia)

        # update
        new_centroids = np.empty_like(centroids)
        for k in range(n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                new_centroids[k] = X[rng.integers(n)]
            else:
                new_centroids[k] = members.mean(axis=0)

        history.append(new_centroids.copy())

        if np.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    return {
        "labels": labels,
        "centroids": centroids,
        "inertia": inertia_history[-1],
        "inertia_history": inertia_history,
        "centroid_history": history,
        "n_iter": len(inertia_history),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
    result = fit_kmeans_with_history(X, n_clusters=4)
    print(f"Converged in {result['n_iter']} iterations, "
          f"inertia={result['inertia']:.1f}")
