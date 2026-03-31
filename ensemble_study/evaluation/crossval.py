"""
Cross-Validation Utilities
============================
RepeatedStratifiedKFold wrapper and multi-seed CV evaluation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from config import CFG
from ensembles.ensemble_factory import build_method


def make_cv(n_splits: int | None = None, n_repeats: int | None = None,
            random_state: int = 42) -> RepeatedStratifiedKFold:
    n_splits = n_splits or CFG.CV_FOLDS
    n_repeats = n_repeats or CFG.CV_REPEATS
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)


def cross_validate_method(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
    random_state: int = 42,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    scoring: List[str] | None = None,
) -> Dict:
    scoring = scoring or ["accuracy", "f1_macro", "balanced_accuracy"]
    clf = build_method(name, n_estimators=n_estimators, random_state=random_state)
    cv = make_cv(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    results = cross_validate(clf, X, y, cv=cv, scoring=scoring,
                              return_train_score=True, n_jobs=-1)
    summary: Dict = {"method": name, "n_estimators": n_estimators,
                      "random_state": random_state}
    for s in scoring:
        key_test = f"test_{s}"
        key_train = f"train_{s}"
        if key_test in results:
            summary[f"cv_{s}_mean"] = float(np.mean(results[key_test]))
            summary[f"cv_{s}_std"] = float(np.std(results[key_test]))
        if key_train in results:
            summary[f"cv_{s}_train_mean"] = float(np.mean(results[key_train]))
    summary["cv_fit_time_mean"] = float(np.mean(results["fit_time"]))
    return summary


def multi_seed_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int] | None = None,
    n_estimators: int = 50,
    **kwargs,
) -> pd.DataFrame:
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    rows = []
    for s in seeds:
        row = cross_validate_method(name, X, y, n_estimators=n_estimators,
                                     random_state=s, **kwargs)
        row["seed"] = s
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    print(cross_validate_method("random_forest", X, y, n_estimators=10))
