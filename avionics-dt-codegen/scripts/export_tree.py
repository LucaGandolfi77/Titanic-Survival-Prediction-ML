#!/usr/bin/env python3
"""
export_tree.py — Train a DecisionTreeClassifier on synthetic engine-sensor
data and export it to the JSON format consumed by the avionics codegen tool.

Usage:
    python3 export_tree.py [--max-depth N] [--n-samples M] [--output FILE]
                           [--export-csv FILE] [--seed S]

Output: a JSON file compatible with codegen_tool.

Copyright 2026 — Avionics ML Codegen Project
SPDX-License-Identifier: MIT
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "T_exhaust",
    "P_inlet",
    "RPM",
    "vibration_x",
    "vibration_y",
    "oil_temp",
    "fuel_flow",
    "EGT",
]

CLASS_NAMES: List[str] = [
    "NOMINAL",
    "WARN_VIBRATION",
    "WARN_THERMAL",
    "FAULT_CRITICAL",
]

DEFAULT_N_SAMPLES: int = 2000
DEFAULT_MAX_DEPTH: int = 6
DEFAULT_SEED: int = 42
DEFAULT_OUTPUT: str = "trained_tree.json"


# ---------------------------------------------------------------------------
# Tree export logic
# ---------------------------------------------------------------------------

def _export_node(
    tree,
    node_id: int,
    feature_names: List[str],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Recursively convert a scikit-learn internal tree node into the
    JSON-serialisable dict format expected by codegen_tool.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The underlying tree structure from a fitted DecisionTreeClassifier.
    node_id : int
        Index of the current node in the sklearn arrays.
    feature_names : list of str
        Human-readable feature names.
    class_names : list of str
        Human-readable class labels.

    Returns
    -------
    dict
        Nested dict representing this subtree.
    """
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # Leaf node.
    if left_child == right_child:  # sklearn uses TREE_LEAF == -1 for both
        class_idx = int(np.argmax(tree.value[node_id]))
        return {
            "node_id": int(node_id),
            "leaf": True,
            "class": class_names[class_idx],
        }

    # Internal node.
    feat_idx = int(tree.feature[node_id])
    threshold = float(tree.threshold[node_id])

    return {
        "node_id": int(node_id),
        "feature": feature_names[feat_idx],
        "threshold": round(threshold, 6),
        "left": _export_node(tree, left_child, feature_names, class_names),
        "right": _export_node(tree, right_child, feature_names, class_names),
    }


def export_tree_to_json(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Export a fitted DecisionTreeClassifier to the codegen JSON format.

    Parameters
    ----------
    clf : DecisionTreeClassifier
        Fitted classifier.
    feature_names : list of str
        Feature names matching the training data columns.
    class_names : list of str
        Class labels in index order.
    output_path : str
        File path for the JSON output.
    """
    tree = clf.tree_
    root_dict = _export_node(tree, 0, feature_names, class_names)

    payload = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "classes": class_names,
        "tree": root_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Exported tree to {output_path}")
    print(f"  Depth       : {clf.get_depth()}")
    print(f"  Leaf count  : {clf.get_n_leaves()}")
    print(f"  Node count  : {clf.tree_.node_count}")


def export_test_csv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Export test data as a CSV for the validation_harness.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : list of str
    class_names : list of str
    output_path : str
    """
    with open(output_path, "w", encoding="utf-8") as f:
        header = ",".join(feature_names) + ",label"
        f.write(header + "\n")
        for i in range(len(X)):
            row = ",".join(f"{v:.6f}" for v in X[i])
            row += "," + class_names[int(y[i])]
            f.write(row + "\n")

    print(f"Exported {len(X)} test samples to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree on synthetic engine data and "
                    "export to JSON for the avionics codegen tool."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum tree depth (default: {DEFAULT_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic samples (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Also export a test CSV for the validation harness.",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Generate synthetic dataset (8 features, 4 classes).
    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=1,
        n_classes=len(CLASS_NAMES),
        random_state=rng,
        flip_y=0.05,
    )

    # Scale features to realistic engine-sensor ranges.
    scales = np.array([
        650.0,   # T_exhaust: ~0–1300 °C
        100.0,   # P_inlet: ~0–200 kPa
        8000.0,  # RPM: ~0–16000
        3.0,     # vibration_x: ~0–6 g
        3.0,     # vibration_y: ~0–6 g
        100.0,   # oil_temp: ~0–200 °C
        60.0,    # fuel_flow: ~0–120 kg/h
        800.0,   # EGT: ~0–1600 °C
    ])
    offsets = scales / 2.0
    X = X * (scales / 4.0) + offsets

    # Train Decision Tree.
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc:.4f}")

    # Export.
    export_tree_to_json(clf, FEATURE_NAMES, CLASS_NAMES, args.output)

    if args.export_csv:
        export_test_csv(X, y, FEATURE_NAMES, CLASS_NAMES, args.export_csv)


if __name__ == "__main__":
    main()
