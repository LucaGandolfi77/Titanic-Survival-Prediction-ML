"""
Cluster factory — build any clustering algorithm from a config dict.

Supported: KMeans, GaussianMixture, AgglomerativeClustering, DBSCAN.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


def build_clusterer(
    algo: str = "kmeans",
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs: Any,
):
    """Instantiate a clustering algorithm by name.

    Parameters
    ----------
    algo : one of 'kmeans', 'gmm', 'agglomerative', 'dbscan'
    n_clusters : number of clusters (ignored by DBSCAN)
    random_state : for reproducibility
    """
    algo = algo.lower()
    if algo == "kmeans":
        return KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
            **kwargs,
        )
    elif algo == "gmm":
        return GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            **kwargs,
        )
    elif algo == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            **kwargs,
        )
    elif algo == "dbscan":
        return DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")


def fit_predict_clusters(
    algo: str,
    X,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs,
):
    """Build, fit, and return cluster labels."""
    model = build_clusterer(algo, n_clusters, random_state, **kwargs)
    if algo == "gmm":
        model.fit(X)
        return model.predict(X), model
    else:
        labels = model.fit_predict(X)
        return labels, model


ALGO_LABELS: Dict[str, str] = {
    "kmeans": "K-Means",
    "gmm": "GMM (EM)",
    "agglomerative": "Agglomerative",
    "dbscan": "DBSCAN",
}


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    for algo in ("kmeans", "gmm", "agglomerative"):
        labels, m = fit_predict_clusters(algo, X, n_clusters=3)
        print(f"{algo:15s}  clusters_found={len(set(labels))}")
