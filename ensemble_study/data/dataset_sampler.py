"""
Dataset Sampler
================
Stratified subsampling for learning-curve experiments.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to exactly n_samples rows."""
    if rng is None:
        rng = np.random.default_rng(0)

    if n_samples >= len(y):
        return X.copy(), y.copy()

    keep_frac = n_samples / len(y)
    drop_frac = 1.0 - keep_frac

    rs = int(rng.integers(0, 2**31))
    try:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, test_size=drop_frac, random_state=rs, stratify=y,
        )
    except ValueError:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, test_size=drop_frac, random_state=rs,
        )

    return X_sub, y_sub


def subsample_series(
    X: np.ndarray,
    y: np.ndarray,
    sizes: List[int],
    rng: np.random.Generator | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Return a list of (X_sub, y_sub, actual_size) for each requested size."""
    if rng is None:
        rng = np.random.default_rng(0)
    results = []
    for s in sizes:
        X_s, y_s = subsample(X, y, s, rng)
        results.append((X_s, y_s, len(y_s)))
    return results


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name

    X, y, _ = get_dataset_by_name("breast_cancer")
    for s in [50, 200, 500]:
        X_s, y_s = subsample(X, y, s, np.random.default_rng(42))
        print(f"target={s}  actual={len(y_s)}  classes={np.unique(y_s, return_counts=True)}")
