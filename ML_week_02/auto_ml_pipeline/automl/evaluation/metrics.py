"""
Metrics helper – surface the right metrics depending on the task.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_proba: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return a dict of metrics appropriate for *task*."""

    if task == "classification":
        is_binary = len(np.unique(y_true)) <= 2
        average = "binary" if is_binary else "weighted"
        m: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None:
            try:
                if is_binary:
                    p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    m["roc_auc"] = float(roc_auc_score(y_true, p))
                    m["log_loss"] = float(log_loss(y_true, p))
                else:
                    m["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
                    m["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass
        return m

    # Regression
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }
