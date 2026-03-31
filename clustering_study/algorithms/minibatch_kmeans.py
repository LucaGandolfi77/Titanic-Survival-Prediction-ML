"""
Mini-Batch K-Means Wrapper
============================
Scalable variant for large datasets.
"""

from __future__ import annotations

from sklearn.cluster import MiniBatchKMeans

from config import CFG


def build_minibatch_kmeans(
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    batch_size: int = 256,
    **kwargs,
) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
