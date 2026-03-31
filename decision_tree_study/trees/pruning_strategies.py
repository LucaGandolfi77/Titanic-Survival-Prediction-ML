"""
Pruning Strategies
==================
High-level functions that encapsulate the logic for each pruning approach:
selecting best hyperparameters via cross-validation and returning a fitted tree.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import CFG
from trees.tree_factory import build_tree


def best_depth_cv(
    X: np.ndarray,
    y: np.ndarray,
    depths: Optional[List[Optional[int]]] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Optional[int], float]:
    """Select best max_depth via stratified k-fold CV."""
    if depths is None:
        depths = CFG.DEPTHS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    best_d, best_score = None, -1.0

    for d in depths:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            tree = build_tree("pre_depth", max_depth=d, random_state=random_state)
            tree.fit(X[train_idx], y[train_idx])
            scores.append(tree.score(X[val_idx], y[val_idx]))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score = mean
            best_d = d

    return best_d, best_score


def best_samples_cv(
    X: np.ndarray,
    y: np.ndarray,
    min_leaf_values: Optional[List[int]] = None,
    min_split_values: Optional[List[int]] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[int, int, float]:
    """Select best min_samples_leaf and min_samples_split via CV."""
    if min_leaf_values is None:
        min_leaf_values = CFG.MIN_SAMPLES_LEAF_VALUES
    if min_split_values is None:
        min_split_values = CFG.MIN_SAMPLES_SPLIT_VALUES

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    best_leaf, best_split, best_score = 1, 2, -1.0

    for ml in min_leaf_values:
        for ms in min_split_values:
            if ms < 2 * ml:
                continue
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                tree = build_tree(
                    "pre_samples",
                    min_samples_leaf=ml,
                    min_samples_split=ms,
                    random_state=random_state,
                )
                tree.fit(X[train_idx], y[train_idx])
                scores.append(tree.score(X[val_idx], y[val_idx]))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score = mean
                best_leaf = ml
                best_split = ms

    return best_leaf, best_split, best_score


def best_ccp_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Select best ccp_alpha via CV on the pruning path.

    Returns (best_alpha, best_score, all_alphas, all_mean_scores).
    """
    base_tree = DecisionTreeClassifier(random_state=random_state)
    base_tree.fit(X, y)
    path = base_tree.cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas

    # Subsample alphas if too many
    if len(alphas) > CFG.ALPHA_N_STEPS:
        idx = np.linspace(0, len(alphas) - 1, CFG.ALPHA_N_STEPS, dtype=int)
        alphas = alphas[idx]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    mean_scores = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            tree = build_tree("ccp", ccp_alpha=alpha, random_state=random_state)
            tree.fit(X[train_idx], y[train_idx])
            scores.append(tree.score(X[val_idx], y[val_idx]))
        mean_scores[i] = np.mean(scores)

    best_idx = int(np.argmax(mean_scores))
    return float(alphas[best_idx]), float(mean_scores[best_idx]), alphas, mean_scores


def best_combined_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Select best max_depth + ccp_alpha via nested CV."""
    best_d, _ = best_depth_cv(
        X, y, depths=[3, 5, 7, 10, 15, None], n_folds=n_folds,
        random_state=random_state,
    )
    best_alpha, best_score, _, _ = best_ccp_alpha_cv(
        X, y, n_folds=n_folds, random_state=random_state,
    )
    return {
        "max_depth": best_d,
        "ccp_alpha": best_alpha,
        "cv_score": best_score,
    }


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name

    X, y, _ = get_dataset_by_name("iris")
    d, s = best_depth_cv(X, y)
    print(f"Best depth: {d}  CV score: {s:.4f}")
    alpha, s2, _, _ = best_ccp_alpha_cv(X, y)
    print(f"Best alpha: {alpha:.6f}  CV score: {s2:.4f}")
