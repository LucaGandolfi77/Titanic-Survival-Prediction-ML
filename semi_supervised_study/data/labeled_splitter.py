"""
Labeled / unlabeled splitter with stratified sampling.

Splits a dataset into a small labeled portion and a larger unlabeled
portion, preserving class proportions via stratification. The unlabeled
set retains its true labels internally (for evaluation) but they are
never exposed to SSL strategies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


def labeled_unlabeled_split(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    labeled_fraction: float,
    rng: np.random.Generator,
) -> Tuple[
    NDArray[np.floating],  # X_labeled
    NDArray[np.integer],   # y_labeled
    NDArray[np.floating],  # X_unlabeled
    NDArray[np.integer],   # y_unlabeled (for evaluation only)
]:
    """Split data into labeled and unlabeled sets with stratification.

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,)
    labeled_fraction : float in (0, 1) — fraction that gets labels
    rng : numpy Generator for reproducibility

    Returns
    -------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled
    """
    n = len(y)
    n_labeled = max(int(n * labeled_fraction), len(np.unique(y)))
    n_labeled = min(n_labeled, n - 1)

    seed = int(rng.integers(0, 2**31))

    X_lab, X_unl, y_lab, y_unl = train_test_split(
        X, y,
        train_size=n_labeled,
        stratify=y,
        random_state=seed,
    )
    return X_lab, y_lab, X_unl, y_unl


if __name__ == "__main__":
    from data.loaders import load_dataset
    X, y, _ = load_dataset("iris")
    rng = np.random.default_rng(42)
    Xl, yl, Xu, yu = labeled_unlabeled_split(X, y, 0.10, rng)
    print(f"Labeled: {len(yl)}  Unlabeled: {len(yu)}")
    print(f"Label ratios: {np.bincount(yl) / len(yl)}")
