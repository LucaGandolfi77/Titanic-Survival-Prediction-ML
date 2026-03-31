"""Confusion-matrix grid — one subplot per method."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from visualization._common import _save, method_label


def plot_confusion_grid(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrices",
    filename: str = "confusion_matrices",
) -> Path:
    methods = list(predictions.keys())
    n = len(methods)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, m in zip(axes_flat, methods):
        cm = confusion_matrix(y_true, predictions[m])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(method_label(m), fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, filename)
