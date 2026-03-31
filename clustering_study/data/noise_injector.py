"""
Noise and Outlier Injection
=============================
Controlled perturbation for robustness experiments.  Always returns copies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def inject_noise(
    X: np.ndarray,
    noise_level: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Add isotropic Gaussian noise scaled by per-feature std.

    Returns (X_noisy, metadata).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_noisy = X.copy().astype(np.float64)
    if noise_level <= 0.0:
        return X_noisy, {"type": "feature_noise", "level": 0.0, "n_affected": 0}

    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    noise = rng.normal(0, noise_level * stds, size=X.shape)
    X_noisy += noise
    return X_noisy, {
        "type": "feature_noise",
        "level": noise_level,
        "n_affected": X.size,
    }


def inject_outliers(
    X: np.ndarray,
    fraction: float,
    rng: np.random.Generator | None = None,
    k: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Replace a fraction of points with extreme outliers at k·σ from mean.

    Returns (X_with_outliers, outlier_mask, metadata).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_out = X.copy().astype(np.float64)
    n = len(X)
    n_outliers = int(round(n * fraction))
    mask = np.zeros(n, dtype=bool)

    if n_outliers == 0:
        return X_out, mask, {"type": "outliers", "fraction": 0.0, "n_outliers": 0}

    idx = rng.choice(n, size=n_outliers, replace=False)
    mask[idx] = True

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0

    for i in idx:
        sign = rng.choice([-1, 1], size=X.shape[1])
        X_out[i] = means + sign * k * stds + rng.normal(0, stds * 0.5)

    return X_out, mask, {
        "type": "outliers",
        "fraction": fraction,
        "n_outliers": int(n_outliers),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 3))
    X_n, meta_n = inject_noise(X, 0.2, rng)
    print(f"Noise: {meta_n}")
    X_o, mask_o, meta_o = inject_outliers(X, 0.1, rng)
    print(f"Outliers: {meta_o}")
