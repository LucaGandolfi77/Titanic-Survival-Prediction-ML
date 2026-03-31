"""
Clustering Stability
=====================
Assess partition reliability via bootstrap resampling.
High stability → the discovered structure is real, not an artefact.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import CFG


def bootstrap_stability(
    X: np.ndarray,
    n_clusters: int,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> Dict:
    """Estimate partition stability via bootstrap ARI.

    Procedure:
    1. Fit K-Means on full data → reference labels.
    2. For each bootstrap:
       a. Resample X with replacement.
       b. Fit K-Means on the bootstrap sample.
       c. Compute ARI between reference labels[bootstrap_idx]
          and bootstrap labels.
    3. Return mean and std of ARI across bootstrap rounds.
    """
    rng = np.random.default_rng(random_state)

    km_ref = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
    ref_labels = km_ref.fit_predict(X)

    ari_scores: List[float] = []
    for b in range(n_bootstrap):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot = X[idx]
        seed = int(rng.integers(100000))
        km_boot = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed)
        boot_labels = km_boot.fit_predict(X_boot)
        ari = adjusted_rand_score(ref_labels[idx], boot_labels)
        ari_scores.append(float(ari))

    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "all_ari": ari_scores,
        "n_clusters": n_clusters,
        "n_bootstrap": n_bootstrap,
    }


def stability_over_k(
    X: np.ndarray,
    k_range: list | tuple | None = None,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> List[Dict]:
    """Compute bootstrap stability for each k in k_range."""
    if k_range is None:
        k_range = list(CFG.K_RANGE)
    results = []
    for k in k_range:
        res = bootstrap_stability(X, k, n_bootstrap=n_bootstrap,
                                  random_state=random_state)
        res["k"] = k
        results.append(res)
    return results


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
    for res in stability_over_k(X, k_range=[2, 3, 4, 5, 6], n_bootstrap=10):
        print(f"  k={res['k']}  ARI={res['mean_ari']:.3f} ± {res['std_ari']:.3f}")
