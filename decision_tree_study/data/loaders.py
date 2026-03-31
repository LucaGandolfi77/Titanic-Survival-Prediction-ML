"""
Dataset Loaders
===============
Load all datasets used in the thesis: three real-world datasets from
sklearn (iris, breast_cancer, wine) and five synthetic datasets generated
via make_classification with increasing structural complexity.

Each loader returns (X, y, dataset_name) with standardised float arrays.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    make_classification,
)


DatasetBundle = Tuple[np.ndarray, np.ndarray, str]


def load_real_datasets() -> List[DatasetBundle]:
    """Return the three real-world benchmark datasets."""
    bundles: List[DatasetBundle] = []
    for loader, name in [
        (load_iris, "iris"),
        (load_breast_cancer, "breast_cancer"),
        (load_wine, "wine"),
    ]:
        data = loader()
        X = data.data.astype(np.float64)
        y = data.target.copy()
        bundles.append((X, y, name))
    return bundles


def _synthetic_spec(variant: int) -> Dict:
    """Return make_classification kwargs for a given complexity variant."""
    specs = {
        0: dict(
            n_samples=2000, n_features=10, n_informative=5,
            n_redundant=2, n_clusters_per_class=1, n_classes=2,
        ),
        1: dict(
            n_samples=2000, n_features=15, n_informative=5,
            n_redundant=5, n_clusters_per_class=2, n_classes=3,
        ),
        2: dict(
            n_samples=3000, n_features=20, n_informative=8,
            n_redundant=5, n_clusters_per_class=2, n_classes=3,
        ),
        3: dict(
            n_samples=3000, n_features=25, n_informative=8,
            n_redundant=7, n_clusters_per_class=3, n_classes=4,
        ),
        4: dict(
            n_samples=5000, n_features=30, n_informative=10,
            n_redundant=10, n_clusters_per_class=3, n_classes=5,
        ),
    }
    return specs[variant]


def load_synthetic_datasets(
    seed: int = 42,
) -> List[DatasetBundle]:
    """Generate five synthetic datasets with increasing complexity."""
    bundles: List[DatasetBundle] = []
    for v in range(5):
        spec = _synthetic_spec(v)
        X, y = make_classification(
            **spec, flip_y=0.0, random_state=seed,
        )
        X = X.astype(np.float64)
        name = f"synth_v{v}"
        bundles.append((X, y, name))
    return bundles


def load_all_datasets(seed: int = 42) -> List[DatasetBundle]:
    """Return all eight datasets (3 real + 5 synthetic)."""
    return load_real_datasets() + load_synthetic_datasets(seed)


def get_dataset_by_name(
    name: str, seed: int = 42
) -> DatasetBundle:
    """Retrieve a single dataset by name."""
    for X, y, n in load_all_datasets(seed):
        if n == name:
            return X, y, n
    raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    for X, y, name in load_all_datasets():
        print(f"{name:20s}  X={X.shape}  classes={len(np.unique(y))}")
