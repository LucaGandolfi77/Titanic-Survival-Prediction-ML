"""
Alignment generator — synthetic datasets with controlled cluster-class NMI.

Generates data where the cluster-class alignment (measured by NMI) can
be tuned continuously from ~0 (clusters independent of classes) to ~1
(clusters perfectly match classes). This is the thesis-defining tool:
it turns cluster-class alignment into a controllable independent variable.

Strategy:
  Start with well-separated blobs (NMI ≈ 1), then progressively add
  Gaussian noise to class-conditional means and inflate intra-class
  standard deviations until the cluster structure no longer reflects
  class boundaries.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def generate_aligned_dataset(
    target_nmi: float,
    n_samples: int = 500,
    n_classes: int = 3,
    n_features: int = 10,
    rng: np.random.Generator | None = None,
    max_attempts: int = 50,
    tol: float = 0.08,
) -> Tuple[NDArray[np.floating], NDArray[np.integer], float]:
    """Generate a synthetic dataset with approximately ``target_nmi``.

    Parameters
    ----------
    target_nmi : float in [0, 1]
    n_samples, n_classes, n_features : dataset shape
    rng : Generator
    max_attempts : how many noise levels to try
    tol : acceptable distance from target NMI

    Returns
    -------
    X, y, actual_nmi
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # well-separated base centres
    centres = np.zeros((n_classes, n_features))
    for c in range(n_classes):
        centres[c, c % n_features] = 6.0 * (c + 1)

    # binary search over noise_scale
    lo, hi = 0.01, 15.0
    best_X, best_y, best_nmi = None, None, -1.0

    for _ in range(max_attempts):
        noise_scale = (lo + hi) / 2.0
        X, y = _make_noisy_blobs(centres, n_samples, noise_scale, n_features, rng)
        X_sc = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=n_classes, n_init=5,
                    random_state=int(rng.integers(0, 2**31)))
        cluster_labels = km.fit_predict(X_sc)
        nmi = normalized_mutual_info_score(y, cluster_labels)

        if abs(nmi - target_nmi) < abs(best_nmi - target_nmi):
            best_X, best_y, best_nmi = X_sc, y, nmi

        if abs(nmi - target_nmi) < tol:
            break

        if nmi > target_nmi:
            lo = noise_scale
        else:
            hi = noise_scale

    return best_X, best_y, best_nmi


def _make_noisy_blobs(
    centres: NDArray,
    n_samples: int,
    noise_scale: float,
    n_features: int,
    rng: np.random.Generator,
) -> Tuple[NDArray, NDArray]:
    n_classes = len(centres)
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    X_parts, y_parts = [], []
    for c in range(n_classes):
        n_c = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.normal(loc=centres[c], scale=noise_scale, size=(n_c, n_features))
        y_c = np.full(n_c, c, dtype=int)
        X_parts.append(X_c)
        y_parts.append(y_c)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


if __name__ == "__main__":
    for target in [0.2, 0.5, 0.8, 1.0]:
        rng = np.random.default_rng(42)
        X, y, actual = generate_aligned_dataset(target, rng=rng)
        print(f"Target NMI={target:.1f}  →  actual={actual:.3f}  shape={X.shape}")
