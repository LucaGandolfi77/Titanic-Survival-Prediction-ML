"""Clustering sub-package: factory, metrics, inspection, optimal K."""

from clustering.cluster_factory import build_clusterer
from clustering.cluster_metrics import (
    compute_cluster_metrics,
    cluster_class_alignment,
    purity_score,
    cluster_class_overlap,
)
from clustering.cluster_inspector import inspect_clusters
from clustering.optimal_k import select_optimal_k

__all__ = [
    "build_clusterer",
    "compute_cluster_metrics",
    "cluster_class_alignment",
    "purity_score",
    "cluster_class_overlap",
    "inspect_clusters",
    "select_optimal_k",
]
