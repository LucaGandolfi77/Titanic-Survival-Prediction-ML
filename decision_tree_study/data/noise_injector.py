"""
Noise Injector
==============
Key methodological contribution of the thesis. Provides controlled,
reproducible injection of label noise (symmetric / asymmetric) and
feature noise (Gaussian / masking) with full metadata logging.

All functions return copies — original arrays are never modified.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np


def inject_label_noise(
    y: np.ndarray,
    noise_rate: float,
    mode: Literal["symmetric", "asymmetric"] = "symmetric",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Flip a fraction of labels to simulate annotation noise.

    Parameters
    ----------
    y : array of shape (n_samples,)
    noise_rate : float in [0, 1)
    mode : "symmetric" (uniform random) or "asymmetric" (i → i+1 mod C)
    rng : numpy Generator for reproducibility

    Returns
    -------
    y_noisy : corrupted labels (copy)
    flipped_mask : boolean mask of affected samples
    metadata : dict with noise parameters and counts
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y_noisy = y.copy()
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    if noise_rate <= 0.0 or n_classes < 2:
        return y_noisy, np.zeros(n, dtype=bool), _meta(
            "label", mode, noise_rate, 0, n
        )

    n_flip = int(round(n * noise_rate))
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    flipped_mask = np.zeros(n, dtype=bool)
    flipped_mask[flip_idx] = True

    if mode == "symmetric":
        for i in flip_idx:
            others = classes[classes != y[i]]
            y_noisy[i] = rng.choice(others)
    elif mode == "asymmetric":
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        for i in flip_idx:
            cur_idx = class_to_idx[y[i]]
            new_idx = (cur_idx + 1) % n_classes
            y_noisy[i] = classes[new_idx]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    actual_flipped = int(np.sum(y_noisy != y))
    return y_noisy, flipped_mask, _meta(
        "label", mode, noise_rate, actual_flipped, n
    )


def inject_feature_noise(
    X: np.ndarray,
    sigma_factor: float = 0.0,
    mask_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict]:
    """Add Gaussian noise and/or random masking to features.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    sigma_factor : noise std as fraction of per-feature std
    mask_rate : fraction of feature values set to zero
    rng : numpy Generator

    Returns
    -------
    X_noisy : corrupted features (copy)
    metadata : dict with noise parameters and counts
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_noisy = X.copy().astype(np.float64)
    n_samples, n_features = X_noisy.shape
    n_affected = 0

    if sigma_factor > 0.0:
        feature_std = np.std(X, axis=0)
        feature_std[feature_std == 0] = 1.0
        noise = rng.normal(0, sigma_factor * feature_std, size=X_noisy.shape)
        X_noisy += noise
        n_affected += n_samples * n_features

    if mask_rate > 0.0:
        mask = rng.random(X_noisy.shape) < mask_rate
        X_noisy[mask] = 0.0
        n_affected += int(mask.sum())

    return X_noisy, {
        "noise_type": "feature",
        "sigma_factor": sigma_factor,
        "mask_rate": mask_rate,
        "n_affected": n_affected,
        "n_total_cells": n_samples * n_features,
    }


def _meta(
    noise_type: str, mode: str, rate: float, n_affected: int, n_total: int
) -> Dict:
    return {
        "noise_type": noise_type,
        "mode": mode,
        "rate": rate,
        "n_affected": n_affected,
        "n_total": n_total,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_n, mask, meta = inject_label_noise(y, 0.3, "symmetric", rng)
    print(f"Label noise: {meta}")
    print(f"Original: {y}")
    print(f"Noisy:    {y_n}")

    X = rng.standard_normal((10, 3))
    X_n, fmeta = inject_feature_noise(X, sigma_factor=0.5, rng=rng)
    print(f"Feature noise: {fmeta}")
