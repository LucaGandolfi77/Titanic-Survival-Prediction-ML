"""
Visualization Utilities
=======================

Matplotlib helpers for training curves, decision boundaries,
confusion matrices, and weight distribution histograms.

All functions accept a ``save_path`` argument (pathlib.Path);
if given the figure is saved, otherwise displayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ────────────────────────────────────────────────────────────────────
# Training curves
# ────────────────────────────────────────────────────────────────────
def plot_training_curves(
    history: dict[str, list[float]],
    save_path: Optional[Path] = None,
    title: str = "Training History",
) -> None:
    """Plot loss and accuracy curves.

    Parameters
    ----------
    history : dict with keys like 'train_loss', 'val_loss',
              'train_acc', 'val_acc'.
    save_path : Path, optional — if given, saves the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="Train Acc")
    if "val_acc" in history and history["val_acc"]:
        ax.plot(history["val_acc"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Decision boundary (2D)
# ────────────────────────────────────────────────────────────────────
def plot_decision_boundary(
    predict_fn,
    X: NDArray,
    Y: NDArray,
    save_path: Optional[Path] = None,
    title: str = "Decision Boundary",
    resolution: int = 200,
) -> None:
    """Plot decision boundary for a 2-feature classifier.

    Parameters
    ----------
    predict_fn : callable — takes X (N, 2) and returns class labels (N,).
    X : ndarray, shape (n_samples, 2)
    Y : ndarray, shape (n_samples,) — integer class labels.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict_fn(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, preds, alpha=0.3, cmap="RdYlBu")
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdYlBu",
        edgecolors="k", s=30, linewidth=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Confusion matrix heatmap
# ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: NDArray,
    class_names: list[str] | None = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : ndarray, shape (n_classes, n_classes)
    class_names : list of str, optional.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title=title,
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Weight distribution histograms
# ────────────────────────────────────────────────────────────────────
def plot_weight_distributions(
    layers: list,
    save_path: Optional[Path] = None,
    title: str = "Weight Distributions",
) -> None:
    """Plot histogram of weights for each trainable layer."""
    trainable = [l for l in layers if l.trainable]
    n = len(trainable)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, layer in zip(axes, trainable):
        W = layer.params["W"].ravel()
        ax.hist(W, bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(repr(layer))
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
