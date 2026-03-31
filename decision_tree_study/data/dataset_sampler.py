"""
Dataset Sampler
===============
Subsample datasets to controlled sizes for the learning-curve experiments.
Uses stratified sampling to preserve class balance at every size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to exactly n_samples rows.

    If n_samples >= len(X), returns copies of the full arrays.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_samples >= len(X):
        return X.copy(), y.copy()

    seed = int(rng.integers(0, 2**31))
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=n_samples,
        stratify=y,
        random_state=seed,
    )
    return X_sub, y_sub


def subsample_series(
    X: np.ndarray,
    y: np.ndarray,
    sizes: list[int],
    rng: Optional[np.random.Generator] = None,
) -> list[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate a list of subsampled datasets at multiple sizes."""
    if rng is None:
        rng = np.random.default_rng(0)
    results = []
    for s in sizes:
        effective = min(s, len(X))
        X_s, y_s = subsample(X, y, effective, rng)
        results.append((X_s, y_s, len(X_s)))
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 4))
    y = np.repeat([0, 1, 2], [200, 200, 100])
    for s in [50, 100, 200, 500]:
        X_s, y_s = subsample(X, y, s, rng)
        print(f"size={s:4d}  got={len(X_s):4d}  classes={np.bincount(y_s)}")
