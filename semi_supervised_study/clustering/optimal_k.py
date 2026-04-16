"""
Optimal K selection via elbow, silhouette sweep, and gap statistic.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _compute_inertias(
    X: NDArray,
    k_range: Sequence[int],
    random_state: int,
) -> Tuple[list, list]:
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        if k >= 2:
            sils.append(float(silhouette_score(X, labels)))
        else:
            sils.append(-1.0)
    return inertias, sils


def select_optimal_k(
    X: NDArray[np.floating],
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    method: str = "silhouette",
    random_state: int = 42,
) -> Dict:
    """Select optimal K using the specified method.

    Parameters
    ----------
    method : 'silhouette', 'elbow', or 'gap'

    Returns
    -------
    dict with 'best_k' and supporting data
    """
    k_range = list(k_range)
    inertias, sils = _compute_inertias(X, k_range, random_state)

    if method == "silhouette":
        valid = [(k, s) for k, s in zip(k_range, sils) if k >= 2]
        best_k = max(valid, key=lambda x: x[1])[0]
        return {"best_k": best_k, "k_range": k_range,
                "silhouettes": sils, "inertias": inertias}

    elif method == "elbow":
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
        else:
            elbow_idx = 0
        best_k = k_range[min(elbow_idx, len(k_range) - 1)]
        return {"best_k": best_k, "k_range": k_range, "inertias": inertias}

    elif method == "gap":
        return _gap_statistic(X, k_range, random_state)

    else:
        raise ValueError(f"Unknown method: {method}")


def _gap_statistic(
    X: NDArray,
    k_range: list,
    random_state: int,
    n_refs: int = 10,
) -> Dict:
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    mins, maxs = X.min(axis=0), X.max(axis=0)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        km.fit(X)
        inertias.append(float(km.inertia_))

    ref_inertias = np.zeros((len(k_range), n_refs))
    for r in range(n_refs):
        X_ref = rng.uniform(mins, maxs, size=(n, d))
        for ki, k in enumerate(k_range):
            km = KMeans(n_clusters=k, n_init=3,
                        random_state=int(rng.integers(0, 2**31)))
            km.fit(X_ref)
            ref_inertias[ki, r] = km.inertia_

    log_inertias = np.log(inertias)
    log_ref = np.log(ref_inertias + 1e-12)
    gaps = log_ref.mean(axis=1) - log_inertias
    sk = log_ref.std(axis=1) * np.sqrt(1 + 1.0 / n_refs)

    best_k = k_range[0]
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            best_k = k_range[i]
            break

    return {"best_k": best_k, "k_range": k_range,
            "gaps": gaps.tolist(), "sk": sk.tolist(),
            "inertias": inertias}


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    for method in ("silhouette", "elbow", "gap"):
        result = select_optimal_k(X, method=method)
        print(f"{method:12s}  best_k={result['best_k']}")
