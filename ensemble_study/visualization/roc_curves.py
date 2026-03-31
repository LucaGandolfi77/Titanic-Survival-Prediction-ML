"""ROC curves — one curve per method, micro/macro average."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from visualization._common import METHOD_COLORS, _save, method_label


def plot_roc_curves(
    probas: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classes: np.ndarray | None = None,
    title: str = "ROC Curves",
    filename: str = "roc_curves",
) -> Path:
    classes = classes if classes is not None else np.unique(y_true)
    n_classes = len(classes)
    binary = n_classes == 2

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    for m, proba in probas.items():
        color = METHOD_COLORS.get(m, None)
        if binary:
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{method_label(m)} (AUC={roc_auc:.3f})",
                    color=color)
        else:
            y_bin = label_binarize(y_true, classes=classes)
            fpr_all, tpr_all = [], []
            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], proba[:, i])
                fpr_all.append(fpr_i)
                tpr_all.append(tpr_i)
            # Macro-average
            all_fpr = np.unique(np.concatenate(fpr_all))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
            mean_tpr /= n_classes
            roc_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr,
                    label=f"{method_label(m)} macro (AUC={roc_auc:.3f})",
                    color=color)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, filename)
