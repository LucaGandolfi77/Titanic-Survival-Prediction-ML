"""
Experiment Utilities
=====================
Shared helpers for running, timing, and collecting metrics.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from algorithms.algorithm_factory import fit_predict_algorithm
from validation.internal_indices import compute_internal_indices
from validation.external_indices import compute_external_indices


def evaluate_clustering(
    name: str,
    X: np.ndarray,
    y_true: np.ndarray | None = None,
    n_clusters: int = 5,
    random_state: int = 42,
    **kwargs,
) -> Dict:
    """Fit a clustering algorithm and compute all validation indices.

    Returns a flat dict suitable for appending to a DataFrame.
    """
    t0 = time.perf_counter()
    labels, alg = fit_predict_algorithm(
        name, X, n_clusters=n_clusters, random_state=random_state, **kwargs,
    )
    fit_time_ms = (time.perf_counter() - t0) * 1000

    k_found = len(np.unique(labels))
    result: Dict = {
        "method": name,
        "n_clusters_requested": n_clusters,
        "n_clusters_found": k_found,
        "random_state": random_state,
        "fit_time_ms": fit_time_ms,
    }

    # internal indices
    internal = compute_internal_indices(X, labels)
    for k, v in internal.items():
        result[f"int_{k}"] = v

    # external indices (only when ground truth is available)
    if y_true is not None:
        external = compute_external_indices(y_true, labels)
        for k, v in external.items():
            result[f"ext_{k}"] = v

    return result
