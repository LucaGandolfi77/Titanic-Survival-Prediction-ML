"""Exp 2 — Macro-F1 and AUC vs class-imbalance ratio."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_imbalance_curves(
    df: pd.DataFrame,
    score_cols: list[str] | None = None,
    ratio_col: str = "imbalance_ratio",
    method_col: str = "method",
    filename: str = "imbalance_curves",
) -> Path:
    score_cols = score_cols or ["test_f1_macro", "test_auc"]
    n = len(score_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, sc in zip(axes, score_cols):
        for m, grp in df.groupby(method_col):
            agg = grp.groupby(ratio_col)[sc].agg(["mean", "std"]).reset_index()
            color = METHOD_COLORS.get(m, None)
            ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["std"],
                        marker="o", label=method_label(m), color=color, capsize=3)
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels(agg[ratio_col], rotation=45)
        ax.set_xlabel("Imbalance Ratio (minority:majority)")
        ax.set_ylabel(sc.replace("_", " ").title())
        ax.set_title(sc.replace("_", " ").title())
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("Class Imbalance Impact", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, filename, subdir="exp2")
