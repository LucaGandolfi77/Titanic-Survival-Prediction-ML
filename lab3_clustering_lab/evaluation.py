"""evaluation.py — Accuracy, confusion matrix, and classes-to-clusters mapping."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def classes_to_clusters(
    labels_true: np.ndarray,
    labels_cluster: np.ndarray,
) -> dict[int, str]:
    """Map each cluster id to the most frequent true class within it.

    Args:
        labels_true: ground-truth class labels (strings).
        labels_cluster: cluster assignments (ints).

    Returns:
        Dict  ``{cluster_id: majority_class_label}``.
    """
    mapping: dict[int, str] = {}
    for cid in sorted(set(int(c) for c in labels_cluster)):
        mask = labels_cluster == cid
        counter = Counter(labels_true[mask])
        majority_class = counter.most_common(1)[0][0]
        mapping[cid] = str(majority_class)
    return mapping


def cluster_accuracy(
    labels_true: np.ndarray,
    labels_cluster: np.ndarray,
    mapping: dict[int, str] | None = None,
) -> tuple[float, np.ndarray]:
    """Compute classification accuracy using a classes-to-clusters mapping.

    Args:
        labels_true: ground-truth class labels.
        labels_cluster: cluster assignments.
        mapping: optional pre-computed mapping.  If *None*, it is derived
            from the data.

    Returns:
        (accuracy, misclassified_mask)
    """
    if mapping is None:
        mapping = classes_to_clusters(labels_true, labels_cluster)

    predicted = np.array([mapping[c] for c in labels_cluster])
    true_str = np.array([str(t) for t in labels_true])
    correct = predicted == true_str
    accuracy = correct.sum() / len(correct)
    return accuracy, ~correct


def confusion_matrix(
    labels_true: np.ndarray,
    labels_cluster: np.ndarray,
    mapping: dict[int, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute a confusion matrix (rows = true classes, cols = predicted).

    Args:
        labels_true: ground-truth class labels.
        labels_cluster: cluster assignments.
        mapping: optional mapping for cluster → class.

    Returns:
        (matrix, class_order) where *class_order* lists unique class labels.
    """
    if mapping is None:
        mapping = classes_to_clusters(labels_true, labels_cluster)

    predicted = np.array([mapping[c] for c in labels_cluster])
    true_str = np.array([str(t) for t in labels_true])

    classes = sorted(set(true_str))
    n_classes = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_str, predicted):
        cm[cls_idx[t], cls_idx[p]] += 1

    return cm, classes


def print_confusion_matrix(cm: np.ndarray, classes: list[str]) -> None:
    """Pretty-print a confusion matrix.

    Args:
        cm: the confusion matrix.
        classes: ordered list of class labels.
    """
    header = "      " + " ".join(f"{c:>5}" for c in classes)
    print(header)
    print("      " + "-" * (6 * len(classes)))
    for i, c in enumerate(classes):
        row = " ".join(f"{cm[i, j]:>5}" for j in range(len(classes)))
        print(f"{c:>5}|{row}")
