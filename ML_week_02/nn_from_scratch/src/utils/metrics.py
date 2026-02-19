"""
Evaluation Metrics
==================

Classification and regression metrics implemented in pure NumPy.
All assume numpy arrays as inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """Classification accuracy.

    .. math::
        \\text{Accuracy} = \\frac{1}{m} \\sum_{i=1}^{m} \\mathbb{1}[\\hat{y}_i = y_i]

    Parameters
    ----------
    y_true : ndarray, shape (n_samples,) — ground-truth labels (integer).
    y_pred : ndarray, shape (n_samples,) — predicted labels (integer).

    Returns
    -------
    float ∈ [0, 1]
    """
    return float(np.mean(y_true.ravel() == y_pred.ravel()))


def precision(y_true: NDArray, y_pred: NDArray, average: str = "macro") -> float:
    """Precision (macro-averaged by default).

    .. math::
        \\text{Precision}_k = \\frac{\\text{TP}_k}{\\text{TP}_k + \\text{FP}_k}

    Parameters
    ----------
    average : 'macro' | 'micro'
    """
    classes = np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))
    precisions = []
    for c in classes:
        tp = np.sum((y_pred.ravel() == c) & (y_true.ravel() == c))
        fp = np.sum((y_pred.ravel() == c) & (y_true.ravel() != c))
        precisions.append(tp / (tp + fp + 1e-15))
    return float(np.mean(precisions))


def recall(y_true: NDArray, y_pred: NDArray, average: str = "macro") -> float:
    """Recall (macro-averaged by default).

    .. math::
        \\text{Recall}_k = \\frac{\\text{TP}_k}{\\text{TP}_k + \\text{FN}_k}
    """
    classes = np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))
    recalls = []
    for c in classes:
        tp = np.sum((y_pred.ravel() == c) & (y_true.ravel() == c))
        fn = np.sum((y_pred.ravel() != c) & (y_true.ravel() == c))
        recalls.append(tp / (tp + fn + 1e-15))
    return float(np.mean(recalls))


def f1_score(y_true: NDArray, y_pred: NDArray, average: str = "macro") -> float:
    """F1 score (harmonic mean of precision and recall).

    .. math::
        F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}
    """
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return float(2 * p * r / (p + r + 1e-15))


def confusion_matrix(y_true: NDArray, y_pred: NDArray, n_classes: int | None = None) -> NDArray:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true    : ndarray, shape (n_samples,)
    y_pred    : ndarray, shape (n_samples,)
    n_classes : int, optional

    Returns
    -------
    cm : ndarray, shape (n_classes, n_classes)
        cm[i, j] = number of samples with true label i predicted as j.
    """
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def mse(y_true: NDArray, y_pred: NDArray) -> float:
    """Mean Squared Error.

    .. math::
        \\text{MSE} = \\frac{1}{m} \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2
    """
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Coefficient of determination R².

    .. math::
        R^2 = 1 - \\frac{\\sum (y - \\hat{y})^2}{\\sum (y - \\bar{y})^2}
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-15))
