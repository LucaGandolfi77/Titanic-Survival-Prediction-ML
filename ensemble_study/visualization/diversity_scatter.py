"""Exp 5 — Diversity (disagreement) vs test F1, coloured homo/hetero."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization._common import _save, method_label


HOMO_METHODS = {"bagging", "random_forest", "adaboost", "gradient_boosting"}
HETERO_METHODS = {"hard_voting", "soft_voting"}


def plot_diversity_scatter(
    df: pd.DataFrame,
    diversity_col: str = "div_disagreement",
    score_col: str = "test_f1_macro",
    method_col: str = "method",
    title: str = "Diversity–Accuracy Trade-off",
    filename: str = "diversity_scatter",
) -> Path:
    df = df.copy()
    df["ensemble_type"] = df[method_col].apply(
        lambda m: "Homogeneous" if m in HOMO_METHODS else "Heterogeneous"
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"Homogeneous": "#1f77b4", "Heterogeneous": "#d62728"}
    for etype, grp in df.groupby("ensemble_type"):
        ax.scatter(grp[diversity_col], grp[score_col],
                   label=etype, color=palette[etype], alpha=0.6, s=40)

    # Annotate method names near cluster centroids
    for m, grp in df.groupby(method_col):
        cx = grp[diversity_col].mean()
        cy = grp[score_col].mean()
        ax.annotate(method_label(m), (cx, cy), fontsize=7,
                    fontweight="bold", alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Diversity (Disagreement)")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    return _save(fig, filename, subdir="exp5")
