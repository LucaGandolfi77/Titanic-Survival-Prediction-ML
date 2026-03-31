"""
Real Datasets
==============
Wrappers around sklearn toy datasets for clustering evaluation.
Returns (X, y_true, name).  y_true is used *only* for external
validation — the algorithms never see it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

DatasetBundle = Tuple[np.ndarray, np.ndarray, str]


def _load_iris(**kwargs) -> DatasetBundle:
    X, y = load_iris(return_X_y=True)
    return X, y, "iris"


def _load_wine(**kwargs) -> DatasetBundle:
    X, y = load_wine(return_X_y=True)
    return X, y, "wine"


def _load_breast_cancer(**kwargs) -> DatasetBundle:
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, "breast_cancer"


ALL_REAL = {
    "iris": _load_iris,
    "wine": _load_wine,
    "breast_cancer": _load_breast_cancer,
}


def get_real_dataset(name: str, **kwargs) -> DatasetBundle:
    if name not in ALL_REAL:
        raise ValueError(f"Unknown real dataset: {name}. "
                         f"Available: {list(ALL_REAL.keys())}")
    return ALL_REAL[name](**kwargs)


if __name__ == "__main__":
    for name, fn in ALL_REAL.items():
        X, y, n = fn()
        print(f"{n:20s}  shape={X.shape}  k={len(np.unique(y))}")
