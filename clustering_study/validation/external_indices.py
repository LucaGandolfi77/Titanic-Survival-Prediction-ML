"""
External Validation Indices
==============================
Metrics that require ground-truth labels, used for benchmarks
on synthetic and labelled real datasets.
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Fowlkes-Mallows Index (FMI)
- Homogeneity, Completeness, V-Measure
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)


def compute_external_indices(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute all external validation indices."""
    hom, comp, v = homogeneity_completeness_v_measure(y_true, labels)
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "fmi": float(fowlkes_mallows_score(y_true, labels)),
        "homogeneity": float(hom),
        "completeness": float(comp),
        "v_measure": float(v),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    X, y = make_blobs(n_samples=300, centers=4, random_state=0)
    labels = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(X)
    ext = compute_external_indices(y, labels)
    for k, v in ext.items():
        print(f"  {k:20s} = {v:.4f}")
