"""
Shared Experiment Utilities
=============================
Common evaluation loop used by all experiment modules.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from config import CFG
from ensembles.diversity_metrics import (
    compute_all_diversity,
    extract_base_predictions,
)
from ensembles.ensemble_factory import build_method

logger = logging.getLogger(__name__)


def evaluate_method(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit a method and compute all recorded metrics.

    Returns a flat dict with test_accuracy, test_f1_macro,
    test_balanced_accuracy, test_auc, train_accuracy,
    overfitting_gap, fit_time_ms, predict_time_ms,
    n_estimators, diversity metrics (if applicable).
    """
    clf = build_method(name, n_estimators=n_estimators, random_state=random_state, **kwargs)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    predict_ms = (time.perf_counter() - t0) * 1000

    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)

    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    # AUC
    test_auc = _safe_auc(clf, X_test, y_test, classes, n_classes)

    # Diversity
    div_metrics: Dict[str, float] = {}
    try:
        base_preds = extract_base_predictions(clf, X_test)
        if len(base_preds) > 1:
            div_metrics = compute_all_diversity(base_preds, y_test)
    except Exception:
        pass

    return {
        "method": name,
        "test_accuracy": test_acc,
        "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "test_auc": test_auc,
        "train_accuracy": train_acc,
        "overfitting_gap": train_acc - test_acc,
        "fit_time_ms": fit_ms,
        "predict_time_ms": predict_ms,
        "n_estimators": n_estimators,
        **{f"div_{k}": v for k, v in div_metrics.items()},
    }


def _safe_auc(clf, X_test, y_test, classes, n_classes) -> float:
    """Compute AUC safely, handling multi-class and missing predict_proba."""
    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_test)
        else:
            return float("nan")

        if n_classes == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            return float(roc_auc_score(y_test, y_prob))
        else:
            y_bin = label_binarize(y_test, classes=classes)
            if y_prob.ndim == 1:
                return float("nan")
            return float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def run_method_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    random_state: int = 42,
    **kwargs: Any,
) -> List[Dict]:
    """Run one method through stratified k-fold and return per-fold results."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        row = evaluate_method(
            name, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            n_estimators=n_estimators, random_state=random_state, **kwargs,
        )
        row["fold"] = fold
        results.append(row)
    return results
