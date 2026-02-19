"""
evaluator.py – Metrics computation for classification & regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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

from .trainer import TrainResult


@dataclass
class EvalResult:
    """Evaluation metrics for a single model."""
    model_key: str
    display_name: str
    task: str
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    report: Optional[str] = None


# ── Classification ────────────────────────────────────────────

def evaluate_classifier(
    train_result: TrainResult,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> EvalResult:
    """Compute standard classification metrics."""
    est = train_result.estimator
    y_pred = est.predict(X_test)

    # Probabilities (if available)
    y_prob = None
    if hasattr(est, "predict_proba"):
        y_prob = est.predict_proba(X_test)

    n_classes = len(np.unique(y_test))
    average = "binary" if n_classes == 2 else "weighted"

    metrics: Dict[str, float] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
    }

    # ROC-AUC
    if y_prob is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
            else:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4
                )
        except Exception:
            metrics["roc_auc"] = float("nan")

    # Log-loss
    if y_prob is not None:
        try:
            metrics["log_loss"] = round(log_loss(y_test, y_prob), 4)
        except Exception:
            pass

    metrics["train_time"] = train_result.train_time_sec

    return EvalResult(
        model_key=train_result.model_key,
        display_name=train_result.display_name,
        task="classification",
        metrics=metrics,
        y_pred=y_pred,
        y_prob=y_prob,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred, zero_division=0),
    )


# ── Regression ────────────────────────────────────────────────

def evaluate_regressor(
    train_result: TrainResult,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> EvalResult:
    """Compute standard regression metrics."""
    y_pred = train_result.estimator.predict(X_test)

    metrics: Dict[str, float] = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mse": round(mean_squared_error(y_test, y_pred), 4),
        "train_time": train_result.train_time_sec,
    }

    return EvalResult(
        model_key=train_result.model_key,
        display_name=train_result.display_name,
        task="regression",
        metrics=metrics,
        y_pred=y_pred,
    )


# ── Dispatcher ────────────────────────────────────────────────

def evaluate(
    train_result: TrainResult,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> EvalResult:
    """Auto-dispatcher based on task type."""
    if train_result.task == "classification":
        return evaluate_classifier(train_result, X_test, y_test)
    return evaluate_regressor(train_result, X_test, y_test)


def evaluate_all(
    train_results: List[TrainResult],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> List[EvalResult]:
    """Evaluate a batch of trained models."""
    return [evaluate(tr, X_test, y_test) for tr in train_results]


def comparison_dataframe(eval_results: List[EvalResult]) -> pd.DataFrame:
    """Build a tidy comparison DataFrame from multiple EvalResults."""
    rows = []
    for er in eval_results:
        row = {"model": er.display_name, **er.metrics}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
