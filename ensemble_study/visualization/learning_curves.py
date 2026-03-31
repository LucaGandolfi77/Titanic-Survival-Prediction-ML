"""Exp 1 — Test F1 vs dataset size (learning curves) per method."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_learning_curves(
    df: pd.DataFrame,
    score_col: str = "test_f1_macro",
    size_col: str = "n_samples",
    method_col: str = "method",
    title: str = "Learning Curves — F1 (macro) vs Dataset Size",
    filename: str = "learning_curves",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, grp in df.groupby(method_col):
        agg = grp.groupby(size_col)[score_col].agg(["mean", "std"]).reset_index()
        color = METHOD_COLORS.get(m, None)
        ax.plot(agg[size_col], agg["mean"], marker="o", label=method_label(m), color=color)
        ax.fill_between(agg[size_col],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, filename, subdir="exp1")
