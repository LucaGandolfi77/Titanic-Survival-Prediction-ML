"""Exp 3 — Accuracy vs label-noise percentage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_noise_curves(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    noise_col: str = "noise_rate",
    method_col: str = "method",
    title: str = "Robustness to Label Noise",
    filename: str = "noise_curves",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, grp in df.groupby(method_col):
        agg = grp.groupby(noise_col)[score_col].agg(["mean", "std"]).reset_index()
        color = METHOD_COLORS.get(m, None)
        ax.plot(agg[noise_col] * 100, agg["mean"], marker="s",
                label=method_label(m), color=color)
        ax.fill_between(agg[noise_col] * 100,
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.12, color=color)
    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, filename, subdir="exp3")
