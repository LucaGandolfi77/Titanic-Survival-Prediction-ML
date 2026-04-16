"""
Cluster-class alignment metrics.

NMI, ARI, Purity, Silhouette, and the custom cluster-class overlap score.
NMI is the primary alignment score used throughout the thesis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def cluster_class_alignment(
    y_true: NDArray[np.integer],
    cluster_labels: NDArray[np.integer],
) -> float:
    """Primary alignment score: NMI between true labels and clusters."""
    return float(normalized_mutual_info_score(y_true, cluster_labels))


def purity_score(
    y_true: NDArray[np.integer],
    cluster_labels: NDArray[np.integer],
) -> float:
    """Cluster purity: fraction of dominant-class samples per cluster."""
    n = len(y_true)
    total = 0
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        classes_in_cluster = y_true[mask]
        counts = np.bincount(classes_in_cluster)
        total += counts.max()
    return total / n


def cluster_class_overlap(
    y_true: NDArray[np.integer],
    cluster_labels: NDArray[np.integer],
) -> float:
    """Custom overlap: mean per-class fraction of samples in a
    cluster where they are majority class."""
    classes = np.unique(y_true)
    per_class_scores = []

    for cls in classes:
        cls_mask = y_true == cls
        cls_clusters = cluster_labels[cls_mask]
        dominant_count = 0

        for c in np.unique(cls_clusters):
            c_mask = cluster_labels == c
            classes_in_c = y_true[c_mask]
            majority_class = np.argmax(np.bincount(classes_in_c))
            if majority_class == cls:
                dominant_count += (cls_clusters == c).sum()

        per_class_scores.append(dominant_count / cls_mask.sum())

    return float(np.mean(per_class_scores))


def compute_cluster_metrics(
    X: NDArray[np.floating],
    y_true: NDArray[np.integer],
    cluster_labels: NDArray[np.integer],
) -> Dict[str, float]:
    """Compute all clustering quality metrics at once."""
    n_unique = len(np.unique(cluster_labels))
    sil = (
        float(silhouette_score(X, cluster_labels))
        if 2 <= n_unique < len(X) else -1.0
    )
    return {
        "nmi": cluster_class_alignment(y_true, cluster_labels),
        "ari": float(adjusted_rand_score(y_true, cluster_labels)),
        "purity": purity_score(y_true, cluster_labels),
        "overlap": cluster_class_overlap(y_true, cluster_labels),
        "silhouette": sil,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    d = load_iris()
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(d.data)
    metrics = compute_cluster_metrics(d.data, d.target, labels)
    for k, v in metrics.items():
        print(f"  {k:12s} = {v:.4f}")
