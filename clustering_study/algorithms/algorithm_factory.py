"""
Algorithm Factory
==================
Build any clustering method from the study by canonical name.
"""

from __future__ import annotations

from typing import Any, Dict

from config import CFG
from algorithms.kmeans import build_kmeans
from algorithms.isodata import build_isodata, ISODATA
from algorithms.minibatch_kmeans import build_minibatch_kmeans
from algorithms.bisecting_kmeans import build_bisecting_kmeans
from algorithms.gmm import build_gmm
from algorithms.adaptive_clustering import build_adaptive, AdaptiveClustering


def build_algorithm(
    name: str,
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    **kwargs: Any,
):
    """Instantiate a clustering algorithm by canonical name.

    Valid names: kmeans, isodata, minibatch_kmeans,
                bisecting_kmeans, gmm, adaptive.
    """
    if name == "kmeans":
        return build_kmeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif name == "isodata":
        return build_isodata(random_state=random_state, **kwargs)
    elif name == "minibatch_kmeans":
        return build_minibatch_kmeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif name == "bisecting_kmeans":
        return build_bisecting_kmeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif name == "gmm":
        return build_gmm(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif name == "adaptive":
        return build_adaptive(n_clusters=n_clusters, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def fit_predict_algorithm(
    name: str,
    X,
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    **kwargs,
):
    """Build, fit, and predict in one call.  Returns labels."""
    alg = build_algorithm(name, n_clusters=n_clusters,
                          random_state=random_state, **kwargs)
    if isinstance(alg, (ISODATA, AdaptiveClustering)):
        alg.fit(X)
        return alg.predict(X), alg
    else:
        labels = alg.fit_predict(X)
        return labels, alg


METHOD_LABELS: Dict[str, str] = {
    "kmeans": "K-Means",
    "isodata": "ISODATA",
    "minibatch_kmeans": "Mini-Batch K-Means",
    "bisecting_kmeans": "Bisecting K-Means",
    "gmm": "GMM (EM)",
    "adaptive": "Adaptive Split/Merge",
}


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=4, random_state=0)
    for name in CFG.METHOD_NAMES:
        labels, alg = fit_predict_algorithm(name, X, n_clusters=4)
        k_found = len(set(labels))
        print(f"{method_label(name):25s}  k_found={k_found}")
