"""
K Selection Methods
====================
Elbow (inertia + kneed), Silhouette, and Gap-statistic wrappers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import CFG
from validation.gap_statistic import gap_statistic


def _elbow_data(
    X: np.ndarray,
    k_range: list | tuple,
    random_state: int = 42,
) -> Dict:
    inertias: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        km.fit(X)
        inertias.append(float(km.inertia_))
    return {"k_range": list(k_range), "inertias": inertias}


def select_k_elbow(
    X: np.ndarray,
    k_range: list | tuple | None = None,
    random_state: int = 42,
) -> Dict:
    """Select k using the elbow / kneed method on WCSS."""
    if k_range is None:
        k_range = list(CFG.K_RANGE)
    data = _elbow_data(X, k_range, random_state)

    try:
        from kneed import KneeLocator
        kl = KneeLocator(data["k_range"], data["inertias"],
                         curve="convex", direction="decreasing")
        best_k = kl.knee if kl.knee is not None else k_range[0]
    except ImportError:
        # fallback: max second derivative
        inertias = np.array(data["inertias"])
        if len(inertias) >= 3:
            d2 = np.diff(inertias, n=2)
            best_k = k_range[int(np.argmax(d2)) + 1]
        else:
            best_k = k_range[0]

    data["best_k"] = int(best_k)
    return data


def select_k_silhouette(
    X: np.ndarray,
    k_range: list | tuple | None = None,
    random_state: int = 42,
) -> Dict:
    """Select k that maximises the mean silhouette score."""
    if k_range is None:
        k_range = list(CFG.K_RANGE)

    scores: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        scores.append(sil)

    best_idx = int(np.argmax(scores))
    return {
        "k_range": list(k_range),
        "silhouettes": scores,
        "best_k": k_range[best_idx],
    }


def select_k_gap(
    X: np.ndarray,
    k_range: list | tuple | None = None,
    n_refs: int = 10,
    random_state: int = 42,
) -> Dict:
    """Select k using the gap statistic."""
    if k_range is None:
        k_range = list(CFG.K_RANGE)
    return gap_statistic(X, k_range=k_range, n_refs=n_refs,
                         random_state=random_state)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
    kr = [2, 3, 4, 5, 6, 7, 8]
    print(f"Elbow  → k={select_k_elbow(X, kr)['best_k']}")
    print(f"Silh.  → k={select_k_silhouette(X, kr)['best_k']}")
    print(f"Gap    → k={select_k_gap(X, kr, n_refs=5)['best_k']}")
