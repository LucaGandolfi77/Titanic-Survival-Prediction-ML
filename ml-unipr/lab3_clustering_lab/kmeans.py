"""kmeans.py — K-Means wrapper with a consistent interface.

Uses ``sklearn.cluster.KMeans`` underneath but exposes a simple API
that the rest of the project can call uniformly.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans as _SKLearnKMeans


class KMeans:
    """Thin wrapper around scikit-learn's KMeans.

    Attributes:
        k: number of clusters.
        centroids: cluster centres after fitting, shape ``(k, n_features)``.
        labels: cluster assignment per sample after fitting.
    """

    def __init__(self, k: int, random_state: int = 42, **kwargs) -> None:
        self.k = k
        self._model = _SKLearnKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            **kwargs,
        )
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means on *X* (ignoring any labels).

        Args:
            X: feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            self
        """
        self._model.fit(X)
        self.centroids = self._model.cluster_centers_
        self.labels = self._model.labels_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest centroid.

        Args:
            X: feature matrix.

        Returns:
            Cluster assignments, shape ``(n_samples,)``.
        """
        return self._model.predict(X)
