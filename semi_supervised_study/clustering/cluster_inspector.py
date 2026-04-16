"""
Cluster inspector — summary statistics for a clustering result.

Cluster size distribution, centroid pairwise distances, intra/inter
cluster variances.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


def inspect_clusters(
    X: NDArray[np.floating],
    labels: NDArray[np.integer],
) -> Dict[str, object]:
    """Return descriptive stats about a clustering partition."""
    unique = np.unique(labels)
    k = len(unique)

    sizes: List[int] = [int(np.sum(labels == c)) for c in unique]
    centroids = np.array([X[labels == c].mean(axis=0) for c in unique])

    # intra-cluster variance per cluster
    intra_variances: List[float] = []
    for c in unique:
        members = X[labels == c]
        intra_variances.append(float(np.mean(np.var(members, axis=0))))

    # inter-cluster: pairwise centroid distances
    inter_dists: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    return {
        "n_clusters": k,
        "sizes": sizes,
        "size_std": float(np.std(sizes)),
        "centroids": centroids,
        "mean_intra_variance": float(np.mean(intra_variances)),
        "mean_inter_distance": float(np.mean(inter_dists)) if inter_dists else 0.0,
        "min_inter_distance": float(np.min(inter_dists)) if inter_dists else 0.0,
    }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, y = make_blobs(n_samples=300, centers=4, random_state=42)
    labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
    info = inspect_clusters(X, labels)
    print(f"K={info['n_clusters']}  sizes={info['sizes']}")
    print(f"intra_var={info['mean_intra_variance']:.3f}  "
          f"inter_dist={info['mean_inter_distance']:.3f}")
