"""
Imbalance-Aware Metrics
========================
Macro-F1, balanced accuracy, AUC-ROC (OVR), G-mean, MCC.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _geometric_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float((y_pred[mask] == c).sum() / mask.sum()))
    if any(r == 0 for r in recalls):
        return 0.0
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def imbalance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    result: Dict[str, float] = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "g_mean": _geometric_mean(y_true, y_pred),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2 and y_proba.ndim == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            elif n_classes > 2 and y_proba.ndim == 2:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            else:
                auc = float("nan")
            result["auc_roc"] = float(auc)
        except (ValueError, IndexError):
            result["auc_roc"] = float("nan")
    else:
        result["auc_roc"] = float("nan")

    return result


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    print(imbalance_metrics(y_true, y_pred))
