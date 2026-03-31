"""
Figure: Confusion Matrices
============================
Plot normalised confusion matrices for selected scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from visualization.style import apply_style, save_fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] | None = None,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix",
    normalize: bool = True,
) -> Path:
    """Plot a single confusion matrix."""
    apply_style()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(5, n_classes), max(4, n_classes - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, filename)


def plot_confusion_grid(
    trees: List[DecisionTreeClassifier],
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    class_names: List[str] | None = None,
    filename: str = "confusion_grid",
) -> Path:
    """Plot confusion matrices for multiple trees side by side."""
    apply_style()
    n = len(trees)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, (tree, label) in enumerate(zip(trees, labels)):
        ax = axes[0, idx]
        y_pred = tree.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        n_classes = cm.shape[0]
        cnames = class_names or [str(i) for i in range(n_classes)]
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=cnames, yticklabels=cnames, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, filename)
