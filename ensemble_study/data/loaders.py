"""
Dataset Loaders
================
Load all datasets used in the study from sklearn.datasets.
Returns standardised tuples (X, y, name) with consistent dtypes.

Datasets:
  - breast_cancer (binary, 30 features)
  - wine (multi-class, 13 features)
  - iris (small multi-class, 4 features)
  - synth_clean (make_classification, low noise)
  - synth_noisy (make_classification, high noise)
  - synth_redundant (make_classification, many redundant features)
  - moons (make_moons, 2D, noise=0.3)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    make_moons,
)

DatasetBundle = Tuple[np.ndarray, np.ndarray, str]


def load_real_datasets() -> List[DatasetBundle]:
    bc = load_breast_cancer()
    wine = load_wine()
    iris = load_iris()
    return [
        (bc.data.astype(np.float64), bc.target, "breast_cancer"),
        (wine.data.astype(np.float64), wine.target, "wine"),
        (iris.data.astype(np.float64), iris.target, "iris"),
    ]


def load_synthetic_datasets(random_state: int = 42) -> List[DatasetBundle]:
    X1, y1 = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=2, n_classes=2, flip_y=0.01, random_state=random_state,
    )
    X2, y2 = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=2, n_classes=2, flip_y=0.15, random_state=random_state + 1,
    )
    X3, y3 = make_classification(
        n_samples=2000, n_features=20, n_informative=5,
        n_redundant=10, n_classes=2, flip_y=0.05, random_state=random_state + 2,
    )
    X4, y4 = make_moons(n_samples=1000, noise=0.3, random_state=random_state)
    return [
        (X1.astype(np.float64), y1, "synth_clean"),
        (X2.astype(np.float64), y2, "synth_noisy"),
        (X3.astype(np.float64), y3, "synth_redundant"),
        (X4.astype(np.float64), y4, "moons"),
    ]


def load_all_datasets() -> List[DatasetBundle]:
    return load_real_datasets() + load_synthetic_datasets()


def get_dataset_by_name(name: str) -> DatasetBundle:
    for ds in load_all_datasets():
        if ds[2] == name:
            return ds
    raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    for X, y, name in load_all_datasets():
        print(f"{name:20s}  X={X.shape}  classes={np.unique(y)}")
