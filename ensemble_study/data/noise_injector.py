"""
Noise Injector
===============
Controlled injection of symmetric label noise for reproducible
experimental conditions.  Always returns copies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def inject_label_noise(
    y: np.ndarray,
    rate: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Flip a fraction of labels uniformly at random (symmetric noise).

    Returns (y_noisy, flipped_mask, metadata).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y_noisy = y.copy()
    n = len(y)
    classes = np.unique(y)

    if rate <= 0.0 or len(classes) < 2:
        return y_noisy, np.zeros(n, dtype=bool), _meta("label_noise", rate, 0, n)

    n_flip = int(round(n * rate))
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    flipped = np.zeros(n, dtype=bool)
    flipped[flip_idx] = True

    for i in flip_idx:
        others = classes[classes != y[i]]
        y_noisy[i] = rng.choice(others)

    actual = int(np.sum(y_noisy != y))
    return y_noisy, flipped, _meta("label_noise", rate, actual, n)


def inject_feature_noise(
    X: np.ndarray,
    sigma_factor: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Add Gaussian noise to features scaled by per-feature std."""
    if rng is None:
        rng = np.random.default_rng(0)

    X_noisy = X.copy().astype(np.float64)
    if sigma_factor <= 0.0:
        return X_noisy, _meta("feature_noise", sigma_factor, 0, X.size)

    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    noise = rng.normal(0, sigma_factor * stds, size=X.shape)
    X_noisy += noise
    return X_noisy, _meta("feature_noise", sigma_factor, X.size, X.size)


def _meta(noise_type: str, rate: float, n_affected: int, n_total: int) -> Dict:
    return {
        "type": noise_type,
        "rate": rate,
        "n_affected": n_affected,
        "n_total": n_total,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = np.array([0, 0, 1, 1, 2, 2] * 10)
    y_n, mask, meta = inject_label_noise(y, 0.2, rng)
    print(f"Label noise: flipped {meta['n_affected']}/{meta['n_total']}")
