"""
Synthetic Dataset Generators
==============================
Each function returns (X, y_true, name) where y_true are the ground-truth
cluster labels used *only* for external validation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

DatasetBundle = Tuple[np.ndarray, np.ndarray, str]


def make_blobs_dataset(
    n_samples: int = 500,
    n_features: int = 2,
    centers: int = 5,
    cluster_std: float = 1.0,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers,
        cluster_std=cluster_std, random_state=random_state,
    )
    return X, y, "blobs"


def make_moons_dataset(
    n_samples: int = 500,
    noise: float = 0.08,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y, "moons"


def make_circles_dataset(
    n_samples: int = 500,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor,
        random_state=random_state,
    )
    return X, y, "circles"


def make_anisotropic_dataset(
    n_samples: int = 500,
    random_state: int = 42,
) -> DatasetBundle:
    """Blobs stretched by a linear transformation."""
    rng = np.random.default_rng(random_state)
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8,
                      random_state=random_state)
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transformation
    return X, y, "anisotropic"


def make_varied_variance_dataset(
    n_samples: int = 500,
    random_state: int = 42,
) -> DatasetBundle:
    """Blobs with very different per-cluster variances."""
    X, y = make_blobs(
        n_samples=n_samples, centers=3,
        cluster_std=[0.5, 2.5, 1.2],
        random_state=random_state,
    )
    return X, y, "varied_variance"


def make_unbalanced_dataset(
    n_samples: int = 600,
    random_state: int = 42,
) -> DatasetBundle:
    """Three clusters with very different sizes."""
    rng = np.random.default_rng(random_state)
    sizes = [n_samples // 6, n_samples // 3, n_samples // 2]
    centers = np.array([[0, 0], [5, 5], [10, 0]])
    parts_X, parts_y = [], []
    for i, (sz, c) in enumerate(zip(sizes, centers)):
        pts = rng.normal(loc=c, scale=0.8, size=(sz, 2))
        parts_X.append(pts)
        parts_y.append(np.full(sz, i))
    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], "unbalanced"


ALL_SYNTHETIC = {
    "blobs": make_blobs_dataset,
    "moons": make_moons_dataset,
    "circles": make_circles_dataset,
    "anisotropic": make_anisotropic_dataset,
    "varied_variance": make_varied_variance_dataset,
    "unbalanced": make_unbalanced_dataset,
}


def get_synthetic_dataset(name: str, **kwargs) -> DatasetBundle:
    if name not in ALL_SYNTHETIC:
        raise ValueError(f"Unknown synthetic dataset: {name}. "
                         f"Available: {list(ALL_SYNTHETIC.keys())}")
    return ALL_SYNTHETIC[name](**kwargs)


if __name__ == "__main__":
    for name, fn in ALL_SYNTHETIC.items():
        X, y, n = fn()
        print(f"{n:20s}  shape={X.shape}  k={len(np.unique(y))}")
