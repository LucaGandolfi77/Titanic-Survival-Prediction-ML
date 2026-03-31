"""Exp 4 — Accuracy vs outlier fraction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_outlier_curves(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    outlier_col: str = "outlier_fraction",
    method_col: str = "method",
    title: str = "Robustness to Outliers",
    filename: str = "outlier_curves",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, grp in df.groupby(method_col):
        agg = grp.groupby(outlier_col)[score_col].agg(["mean", "std"]).reset_index()
        color = METHOD_COLORS.get(m, None)
        ax.plot(agg[outlier_col] * 100, agg["mean"], marker="^",
                label=method_label(m), color=color)
        ax.fill_between(agg[outlier_col] * 100,
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.12, color=color)
    ax.set_xlabel("Outlier Fraction (%)")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, filename, subdir="exp4")
