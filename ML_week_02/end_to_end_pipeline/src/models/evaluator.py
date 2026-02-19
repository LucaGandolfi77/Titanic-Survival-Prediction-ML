"""
evaluator.py — Metrics Computation
===================================
Computes a standard set of classification metrics and optionally
generates plots (confusion matrix, ROC curve) for MLflow logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("titanic_mlops.evaluator")


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    model : estimator
        A fitted sklearn-compatible classifier.
    X : ndarray
        Feature matrix.
    y : ndarray
        True labels.

    Returns
    -------
    dict
        Metric name → value mapping.
    """
    y_pred = model.predict(X)

    # Probabilities for ROC-AUC (if the model supports it)
    try:
        y_proba = model.predict_proba(X)[:, 1]
        roc_auc = float(roc_auc_score(y, y_proba))
    except (AttributeError, IndexError):
        roc_auc = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def generate_report(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate a human-readable classification report.

    Parameters
    ----------
    model : estimator
        Fitted classifier.
    X : ndarray
        Feature matrix.
    y : ndarray
        True labels.
    output_dir : Path, optional
        If given, also save the report as a text file.

    Returns
    -------
    str
        The classification report text.
    """
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=["Not Survived", "Survived"])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "classification_report.txt"
        report_path.write_text(report)
        logger.info("Classification report saved → %s", report_path)

    return report


def plot_confusion_matrix(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    model : estimator
    X, y : ndarray
    output_dir : Path, optional

    Returns
    -------
    Path or None
        Path to the saved figure, if output_dir was given.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Not Survived", "Survived"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Print values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / "confusion_matrix.png"
        fig.savefig(saved_path, dpi=150)
        logger.info("Confusion matrix saved → %s", saved_path)

    plt.close(fig)
    return saved_path
