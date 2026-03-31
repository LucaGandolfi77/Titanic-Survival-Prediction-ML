"""
Cross-Validation Utilities
===========================
Stratified K-Fold evaluation repeated across multiple seeds.
Returns tidy DataFrames ready for statistical tests.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import CFG


def multi_seed_cv(
    X: np.ndarray,
    y: np.ndarray,
    tree_kwargs: Dict | None = None,
    n_folds: int | None = None,
    seeds: List[int] | None = None,
) -> pd.DataFrame:
    """
    Run StratifiedKFold for each seed and return per-fold results.

    Returns DataFrame with columns: seed, fold, train_accuracy,
    val_accuracy, train_f1, val_f1, n_leaves, tree_depth.
    """
    n_folds = n_folds or CFG.CV_FOLDS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    tree_kwargs = tree_kwargs or {}
    rows: List[Dict] = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            tree = DecisionTreeClassifier(random_state=seed, **tree_kwargs)
            tree.fit(X_tr, y_tr)

            y_tr_pred = tree.predict(X_tr)
            y_va_pred = tree.predict(X_va)

            rows.append({
                "seed": seed,
                "fold": fold_idx,
                "train_accuracy": accuracy_score(y_tr, y_tr_pred),
                "val_accuracy": accuracy_score(y_va, y_va_pred),
                "train_f1": f1_score(y_tr, y_tr_pred, average="weighted", zero_division=0),
                "val_f1": f1_score(y_va, y_va_pred, average="weighted", zero_division=0),
                "n_leaves": tree.get_n_leaves(),
                "tree_depth": tree.get_depth(),
            })

    return pd.DataFrame(rows)


def cv_summary(cv_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate a multi_seed_cv result into mean ± std."""
    acc = cv_df["val_accuracy"]
    return {
        "mean_val_accuracy": float(acc.mean()),
        "std_val_accuracy": float(acc.std()),
        "mean_val_f1": float(cv_df["val_f1"].mean()),
        "mean_train_accuracy": float(cv_df["train_accuracy"].mean()),
        "mean_n_leaves": float(cv_df["n_leaves"].mean()),
        "mean_tree_depth": float(cv_df["tree_depth"].mean()),
        "n_evaluations": len(cv_df),
    }
