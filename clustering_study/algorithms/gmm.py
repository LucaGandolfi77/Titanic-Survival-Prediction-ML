"""
Gaussian Mixture Model Wrapper
=================================
Soft (probabilistic) clustering baseline.
"""

from __future__ import annotations

from sklearn.mixture import GaussianMixture

from config import CFG


def build_gmm(
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    covariance_type: str = "full",
    **kwargs,
) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=5,
        max_iter=200,
        random_state=random_state,
    )
