"""
Tree Metrics
============
Compute all metrics used in the experimental analyses: accuracy, F1,
structural measures (depth, leaves, nodes), interpretability score,
and the overfitting gap.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def compute_metrics(
    tree: DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute the full metric suite for a fitted tree.

    Returns a flat dictionary suitable for DataFrame row construction.
    """
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    avg = "binary" if n_classes == 2 else "macro"

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)
    test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_f1 = f1_score(y_train, train_pred, average=avg, zero_division=0)
    val_f1 = f1_score(y_val, val_pred, average=avg, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average=avg, zero_division=0)

    depth = tree.get_depth()
    n_leaves = tree.get_n_leaves()
    n_nodes = tree.tree_.node_count

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "tree_depth": depth,
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
        "interpretability_score": 1.0 / (1.0 + n_leaves),
        "overfitting_gap": train_acc - test_acc,
    }


def timed_fit(
    tree: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Fit the tree and return training time in milliseconds."""
    t0 = time.perf_counter()
    tree.fit(X, y)
    return (time.perf_counter() - t0) * 1000.0


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name
    from sklearn.model_selection import train_test_split
    from trees.tree_factory import build_tree

    X, y, _ = get_dataset_by_name("iris")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = build_tree("none", random_state=42)
    ms = timed_fit(tree, X_tr, y_tr)
    m = compute_metrics(tree, X_tr, y_tr, X_te, y_te, X_te, y_te)
    print(f"Fit time: {ms:.2f} ms")
    for k, v in m.items():
        print(f"  {k}: {v}")
