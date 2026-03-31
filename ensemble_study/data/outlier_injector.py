"""
Outlier Injector
=================
Inject synthetic feature-space outliers by replacing a fraction of
samples with extreme values drawn from N(mean + k*std, std) with k=4.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def inject_outliers(
    X: np.ndarray,
    fraction: float,
    rng: np.random.Generator | None = None,
    k: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Replace *fraction* of rows with extreme outlier values.

    Returns (X_contaminated, outlier_mask, metadata).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_out = X.copy().astype(np.float64)
    n = len(X_out)

    if fraction <= 0.0:
        return X_out, np.zeros(n, dtype=bool), _meta(fraction, 0, n)

    n_outliers = max(1, int(round(n * fraction)))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[outlier_idx] = True

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0

    for i in outlier_idx:
        signs = rng.choice([-1, 1], size=X.shape[1])
        X_out[i] = rng.normal(means + signs * k * stds, stds)

    return X_out, mask, _meta(fraction, int(mask.sum()), n)


def _meta(fraction: float, n_outliers: int, n_total: int) -> Dict:
    return {
        "type": "outlier",
        "fraction": fraction,
        "n_outliers": n_outliers,
        "n_total": n_total,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    X_out, mask, meta = inject_outliers(X, 0.1, rng)
    print(f"Injected {meta['n_outliers']} outliers into {meta['n_total']} samples")
    print(f"Max value before: {X.max():.2f}, after: {X_out.max():.2f}")
