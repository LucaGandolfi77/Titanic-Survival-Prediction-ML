"""
Gap Statistic
==============
Tibshirani, Walther & Hastie (2001).

Compare log(WCSS_k) on the data to the expected log(WCSS_k) under a
uniform reference distribution.  Choose k that maximises the gap.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from config import CFG


def _reference_wcss(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_refs: int = 10,
) -> Tuple[float, float]:
    """Generate n_refs uniform reference datasets and compute mean/std of log(WCSS)."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    log_wcss_refs = []
    for _ in range(n_refs):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        km = KMeans(n_clusters=k, n_init=3, max_iter=100,
                    random_state=int(rng.integers(10000)))
        km.fit(X_ref)
        log_wcss_refs.append(np.log(km.inertia_ + 1e-15))
    return float(np.mean(log_wcss_refs)), float(np.std(log_wcss_refs))


def gap_statistic(
    X: np.ndarray,
    k_range: tuple | list | None = None,
    n_refs: int = 10,
    random_state: int = 42,
) -> Dict:
    """Compute gap statistic for a range of k values.

    Returns dict with:
        gaps:     list of gap values
        sk:       list of adjusted std-dev
        k_range:  list of k values tested
        best_k:   optimal k (first k where gap_k >= gap_{k+1} - s_{k+1})
        wcss:     list of WCSS on actual data
    """
    if k_range is None:
        k_range = list(CFG.K_RANGE)
    k_range = sorted(k_range)

    rng = np.random.default_rng(random_state)
    gaps: List[float] = []
    sk: List[float] = []
    wcss: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        km.fit(X)
        log_wk = np.log(km.inertia_ + 1e-15)
        wcss.append(float(km.inertia_))

        ref_mean, ref_std = _reference_wcss(X, k, rng, n_refs)
        gap = ref_mean - log_wk
        s = ref_std * np.sqrt(1 + 1 / n_refs)
        gaps.append(float(gap))
        sk.append(float(s))

    # selection: first k where gap_k >= gap_{k+1} - s_{k+1}
    best_k = k_range[-1]
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            best_k = k_range[i]
            break

    return {
        "gaps": gaps,
        "sk": sk,
        "k_range": k_range,
        "best_k": best_k,
        "wcss": wcss,
    }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
    result = gap_statistic(X, k_range=[2, 3, 4, 5, 6, 7, 8])
    print(f"Best k (gap): {result['best_k']}")
    for k, g, s in zip(result["k_range"], result["gaps"], result["sk"]):
        print(f"  k={k}  gap={g:.3f}  sk={s:.3f}")
