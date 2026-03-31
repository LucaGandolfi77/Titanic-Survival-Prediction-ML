"""
Preprocessing Utilities
========================
Scaling and dimensionality reduction helpers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_data(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """PCA dimensionality reduction for visualization or analysis."""
    pca = PCA(n_components=min(n_components, X.shape[1]),
              random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


if __name__ == "__main__":
    from data.real_datasets import get_real_dataset
    X, y, name = get_real_dataset("wine")
    X_s, _ = scale_data(X)
    X_r, pca = reduce_dimensions(X_s)
    print(f"{name}: {X.shape} → scaled → PCA {X_r.shape}  "
          f"var_explained={pca.explained_variance_ratio_.sum():.3f}")
