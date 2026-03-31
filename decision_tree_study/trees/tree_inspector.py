"""
Tree Inspector
==============
Extract interpretable information from fitted decision trees:
decision rules, depth distribution of splits, node impurity statistics,
and feature importance rankings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


def extract_rules(tree: DecisionTreeClassifier, feature_names: List[str] | None = None) -> str:
    """Return a text representation of all decision rules."""
    return export_text(tree, feature_names=feature_names, max_depth=20)


def depth_distribution(tree: DecisionTreeClassifier) -> Dict[int, int]:
    """Count the number of nodes at each depth level."""
    t = tree.tree_
    depths: Dict[int, int] = {}

    def _walk(node: int, depth: int) -> None:
        depths[depth] = depths.get(depth, 0) + 1
        left, right = t.children_left[node], t.children_right[node]
        if left != -1:
            _walk(left, depth + 1)
        if right != -1:
            _walk(right, depth + 1)

    _walk(0, 0)
    return depths


def node_impurity_stats(tree: DecisionTreeClassifier) -> Dict[str, float]:
    """Summary statistics of Gini impurity across all nodes."""
    imp = tree.tree_.impurity
    return {
        "mean_impurity": float(np.mean(imp)),
        "std_impurity": float(np.std(imp)),
        "max_impurity": float(np.max(imp)),
        "min_impurity": float(np.min(imp)),
        "median_impurity": float(np.median(imp)),
    }


def feature_importance_ranking(
    tree: DecisionTreeClassifier,
    feature_names: List[str] | None = None,
) -> List[Tuple[str, float]]:
    """Return features sorted by importance (descending)."""
    importances = tree.feature_importances_
    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    paired = list(zip(feature_names, importances))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name
    from sklearn.model_selection import train_test_split
    from trees.tree_factory import build_tree

    X, y, _ = get_dataset_by_name("iris")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = build_tree("none", random_state=42)
    tree.fit(X_tr, y_tr)

    print("=== Rules (first 20 lines) ===")
    rules = extract_rules(tree)
    for line in rules.split("\n")[:20]:
        print(line)

    print("\n=== Depth Distribution ===")
    print(depth_distribution(tree))

    print("\n=== Impurity Stats ===")
    print(node_impurity_stats(tree))

    print("\n=== Feature Importance ===")
    for name, imp in feature_importance_ranking(tree):
        print(f"  {name}: {imp:.4f}")
