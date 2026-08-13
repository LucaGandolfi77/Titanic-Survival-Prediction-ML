"""utils.py — Shared helpers: metrics printing, plotting, ARFF writing."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def print_metrics(y_true, y_pred, label: str = "") -> float:
    """Print accuracy, classification report, and confusion matrix.

    Returns the accuracy value.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  [{label}] Accuracy: {acc:.2%}")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    header = "  " + "".join(f"{l:>6}" for l in labels)
    print("  Confusion matrix:")
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels[i]:>3}|" + "".join(f"{v:>6}" for v in row))
    return acc


def plot_confusion_matrix(y_true, y_pred, title: str,
                          filepath: str) -> None:
    """Save a confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {filepath}")


def plot_decision_boundary(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    filepath: str | None = None,
    ax: plt.Axes | None = None,
    label_map: dict | None = None,
    resolution: float = 0.05,
) -> plt.Axes:
    """2-D decision-boundary plot for classifiers with 2 features.

    If *ax* is given, draws into that axes; otherwise creates a figure.
    *label_map* converts string labels → ints for colouring.
    """
    if label_map is None:
        unique = sorted(set(y))
        label_map = {l: i for i, l in enumerate(unique)}

    y_int = np.array([label_map[l] for l in y])

    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z_int = np.array([label_map.get(l, 0) for l in Z])
    Z_int = Z_int.reshape(xx.shape)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.contourf(xx, yy, Z_int, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_int, cmap="coolwarm",
                         edgecolors="k", s=20, linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if own_fig and filepath:
        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] Saved: {filepath}")

    return ax
