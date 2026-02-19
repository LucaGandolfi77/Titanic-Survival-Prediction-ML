"""
Data Utilities
==============

Helpers for batching, shuffling, splitting, and encoding datasets
using only NumPy.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray


# ────────────────────────────────────────────────────────────────────
# Batch generator
# ────────────────────────────────────────────────────────────────────
class BatchGenerator:
    """Iterate over (X, Y) in mini-batches.

    Parameters
    ----------
    X          : ndarray, shape (n_samples, ...)
    Y          : ndarray, shape (n_samples, ...)
    batch_size : int — number of samples per batch.
    shuffle    : bool — shuffle before iterating.
    rng        : Generator, optional — for reproducibility.

    Yields
    ------
    (X_batch, Y_batch) : tuple of ndarrays
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        batch_size: int = 32,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = rng or np.random.default_rng()

    def __iter__(self) -> Iterator[tuple[NDArray, NDArray]]:
        n = self.X.shape[0]
        indices = np.arange(n)

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            yield self.X[idx], self.Y[idx]

    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size))


# ────────────────────────────────────────────────────────────────────
# Shuffle
# ────────────────────────────────────────────────────────────────────
def shuffle_data(
    X: NDArray,
    Y: NDArray,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Shuffle X and Y in unison.

    Parameters
    ----------
    X : ndarray, shape (n_samples, ...)
    Y : ndarray, shape (n_samples, ...)

    Returns
    -------
    (X_shuffled, Y_shuffled)
    """
    if rng is None:
        rng = np.random.default_rng()

    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], Y[idx]


# ────────────────────────────────────────────────────────────────────
# Train / test split
# ────────────────────────────────────────────────────────────────────
def train_test_split(
    X: NDArray,
    Y: NDArray,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split data into train and test sets.

    Parameters
    ----------
    test_size : float ∈ (0, 1) — fraction of data used for testing.

    Returns
    -------
    (X_train, X_test, Y_train, Y_test)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]

    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


# ────────────────────────────────────────────────────────────────────
# One-hot encoding
# ────────────────────────────────────────────────────────────────────
def one_hot_encode(
    labels: NDArray,
    n_classes: int | None = None,
) -> NDArray:
    """Convert integer labels to one-hot vectors.

    Parameters
    ----------
    labels    : ndarray, shape (n_samples,) — integer class labels.
    n_classes : int, optional — number of classes (auto-detected if None).

    Returns
    -------
    one_hot : ndarray, shape (n_samples, n_classes)
    """
    labels = labels.astype(int).ravel()
    if n_classes is None:
        n_classes = int(labels.max()) + 1

    one_hot: NDArray = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


# ────────────────────────────────────────────────────────────────────
# Normalization helpers
# ────────────────────────────────────────────────────────────────────
def normalize(X: NDArray, axis: int = 0) -> tuple[NDArray, NDArray, NDArray]:
    """Z-score normalization: (X − μ) / σ.

    Returns
    -------
    (X_norm, mean, std)
    """
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std


def min_max_scale(X: NDArray) -> NDArray:
    """Scale features to [0, 1]."""
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    return (X - x_min) / (x_max - x_min + 1e-8)
