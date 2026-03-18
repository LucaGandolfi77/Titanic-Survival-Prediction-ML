"""xmeans.py — X-Means implementation using BIC-based cluster splitting.

A cluster is split when the two child centroids produced by local
KMeans(k=2) yield a better BIC than the parent.  Splitting continues
until no improvement is found or *max_clusters* is reached.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans as _SKLearnKMeans


def _bic_single(X: np.ndarray) -> float:
    """BIC of a single Gaussian cluster centred on the sample mean.

    BIC = -2·L + p·ln(n)  where L is the log-likelihood of a
    spherical Gaussian and p is the number of free parameters.
    """
    n, d = X.shape
    if n <= 1:
        return 0.0
    centre = X.mean(axis=0)
    var = np.sum((X - centre) ** 2) / max(n, 1)
    var = max(var, 1e-12)
    # log-likelihood
    ll = -n / 2.0 * np.log(2 * np.pi) - n * d / 2.0 * np.log(var / d) \
         - (n - 1) / 2.0
    # free params: d (mean) + 1 (variance)
    p = d + 1
    return -2.0 * ll + p * np.log(n)


def _bic_two(X: np.ndarray, labels: np.ndarray) -> float:
    """Sum of BIC scores for two child clusters."""
    total = 0.0
    for c in (0, 1):
        Xc = X[labels == c]
        if len(Xc) == 0:
            continue
        total += _bic_single(Xc)
    return total


class XMeans:
    """X-Means clustering via BIC-based splitting.

    Attributes:
        min_clusters: minimum number of clusters.
        max_clusters: maximum number of clusters.
        centroids: final cluster centres.
        labels: cluster assignment per training sample.
        k: actual number of clusters found.
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 20,
        random_state: int = 42,
    ) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.k: int = 0

    def fit(self, X: np.ndarray) -> "XMeans":
        """Run X-Means on *X*.

        Args:
            X: feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            self
        """
        n, d = X.shape

        # Start with min_clusters via regular KMeans
        init_km = _SKLearnKMeans(
            n_clusters=self.min_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(X)
        centroids = list(init_km.cluster_centers_)
        labels = init_km.labels_.copy()

        rng = np.random.RandomState(self.random_state)

        changed = True
        while changed and len(centroids) < self.max_clusters:
            changed = False
            new_centroids: list[np.ndarray] = []
            new_labels = np.full(n, -1, dtype=int)

            for ci, centre in enumerate(centroids):
                mask = labels == ci
                Xc = X[mask]
                if len(Xc) < 2:
                    new_centroids.append(centre)
                    idx = len(new_centroids) - 1
                    new_labels[mask] = idx
                    continue

                bic_parent = _bic_single(Xc)

                if len(centroids) + 1 > self.max_clusters:
                    # Can't split further
                    new_centroids.append(centre)
                    idx = len(new_centroids) - 1
                    new_labels[mask] = idx
                    continue

                # Try splitting the cluster
                local_km = _SKLearnKMeans(
                    n_clusters=2,
                    random_state=rng.randint(0, 2**31),
                    n_init=5,
                ).fit(Xc)
                bic_children = _bic_two(Xc, local_km.labels_)

                if bic_children < bic_parent:
                    # Split accepted
                    idx0 = len(new_centroids)
                    new_centroids.append(local_km.cluster_centers_[0])
                    new_centroids.append(local_km.cluster_centers_[1])
                    local_labels = local_km.labels_
                    indices = np.where(mask)[0]
                    for li, gi in enumerate(indices):
                        new_labels[gi] = idx0 + local_labels[li]
                    changed = True
                else:
                    new_centroids.append(centre)
                    idx = len(new_centroids) - 1
                    new_labels[mask] = idx

            centroids = new_centroids
            labels = new_labels

            # Trim to max
            if len(centroids) > self.max_clusters:
                break

        self.centroids = np.array(centroids)
        self.labels = labels
        self.k = len(centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to the nearest centroid.

        Args:
            X: feature matrix.

        Returns:
            Cluster assignments.
        """
        if self.centroids is None:
            raise RuntimeError("Model not fitted yet.")
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2,
        )
        return np.argmin(dists, axis=1)
