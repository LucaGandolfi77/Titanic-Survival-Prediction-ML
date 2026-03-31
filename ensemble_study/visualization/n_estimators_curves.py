"""Exp 6 — Accuracy vs number of estimators."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_n_estimators_curves(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    ne_col: str = "n_estimators",
    method_col: str = "method",
    title: str = "Accuracy vs Number of Estimators",
    filename: str = "n_estimators_curves",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, grp in df.groupby(method_col):
        agg = grp.groupby(ne_col)[score_col].agg(["mean", "std"]).reset_index()
        color = METHOD_COLORS.get(m, None)
        ax.plot(agg[ne_col], agg["mean"], marker="D",
                label=method_label(m), color=color)
        ax.fill_between(agg[ne_col],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.12, color=color)
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, filename, subdir="exp6")
