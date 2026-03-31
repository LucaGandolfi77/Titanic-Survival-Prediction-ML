"""
Bisecting K-Means Wrapper
===========================
Top-down hierarchical variant.
"""

from __future__ import annotations

from sklearn.cluster import BisectingKMeans

from config import CFG


def build_bisecting_kmeans(
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    **kwargs,
) -> BisectingKMeans:
    return BisectingKMeans(
        n_clusters=n_clusters,
        n_init=3,
        max_iter=300,
        random_state=random_state,
    )
