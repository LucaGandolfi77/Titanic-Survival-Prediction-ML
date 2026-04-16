"""
Dataset loaders for the semi-supervised study.

Provides a unified interface to load real (sklearn) and synthetic datasets.
Each loader returns (X, y, dataset_name) with X already scaled to unit
variance via StandardScaler.
"""

from __future__ import annotations

from typing import Dict, Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    make_blobs,
    make_classification,
)
from sklearn.preprocessing import StandardScaler

DatasetBundle = Tuple[NDArray[np.floating], NDArray[np.integer], str]


def _load_iris() -> DatasetBundle:
    d = load_iris()
    X = StandardScaler().fit_transform(d.data)
    return X, d.target, "iris"


def _load_wine() -> DatasetBundle:
    d = load_wine()
    X = StandardScaler().fit_transform(d.data)
    return X, d.target, "wine"


def _load_breast_cancer() -> DatasetBundle:
    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    return X, d.target, "breast_cancer"


def _load_make_blobs(
    n_samples: int = 500,
    n_features: int = 10,
    centers: int = 4,
    cluster_std: float = 1.0,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, "make_blobs"


def _load_make_classification(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 3,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, "make_classification"


REAL_LOADERS: Dict[str, Callable[..., DatasetBundle]] = {
    "iris": _load_iris,
    "wine": _load_wine,
    "breast_cancer": _load_breast_cancer,
}

SYNTH_LOADERS: Dict[str, Callable[..., DatasetBundle]] = {
    "make_blobs": _load_make_blobs,
    "make_classification": _load_make_classification,
}

ALL_DATASETS: Dict[str, Callable[..., DatasetBundle]] = {
    **REAL_LOADERS,
    **SYNTH_LOADERS,
}


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return ALL_DATASETS[name](**kwargs)


if __name__ == "__main__":
    for name in ALL_DATASETS:
        X, y, dname = load_dataset(name)
        print(f"{dname:25s}  X={X.shape}  classes={len(np.unique(y))}")
