"""
metadata.py – Extract descriptive metadata from any sklearn-compatible estimator.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def extract_model_info(estimator: BaseEstimator) -> Dict[str, Any]:
    """Return a descriptive dict about the estimator's type & params."""
    info: Dict[str, Any] = {
        "class": type(estimator).__name__,
        "module": type(estimator).__module__,
        "is_classifier": hasattr(estimator, "predict_proba") or hasattr(estimator, "classes_"),
        "params": estimator.get_params(),
    }

    # Feature count
    if hasattr(estimator, "n_features_in_"):
        info["n_features"] = int(estimator.n_features_in_)
    if hasattr(estimator, "feature_names_in_"):
        info["feature_names"] = list(estimator.feature_names_in_)
    if hasattr(estimator, "classes_"):
        info["classes"] = [str(c) for c in estimator.classes_]
        info["n_classes"] = len(estimator.classes_)

    # Tree count (ensemble models)
    if hasattr(estimator, "n_estimators"):
        info["n_estimators"] = estimator.n_estimators

    return info


def compute_performance(
    estimator: BaseEstimator,
    X_test,
    y_test,
) -> Dict[str, float]:
    """Compute standard classification or regression metrics."""
    y_pred = estimator.predict(X_test)
    is_clf = hasattr(estimator, "predict_proba") or hasattr(estimator, "classes_")

    if is_clf:
        n_classes = len(np.unique(y_test))
        avg = "binary" if n_classes == 2 else "weighted"
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }
        if hasattr(estimator, "predict_proba"):
            try:
                proba = estimator.predict_proba(X_test)
                if n_classes == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_test, proba[:, 1]), 4)
                else:
                    metrics["roc_auc"] = round(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 4
                    )
            except Exception:
                pass
    else:
        metrics = {
            "r2": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        }
    return metrics
