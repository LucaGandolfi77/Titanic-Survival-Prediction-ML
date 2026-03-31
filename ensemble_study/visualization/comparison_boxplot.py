"""Comparison box-plots — all 8 methods side by side."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from visualization._common import PALETTE, _save, method_label


def plot_comparison_boxplot(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    method_col: str = "method",
    title: str = "Method Comparison",
    filename: str = "comparison_boxplot",
) -> Path:
    df = df.copy()
    df["label"] = df[method_col].map(method_label)
    order = (df.groupby("label")[score_col]
             .mean()
             .sort_values(ascending=False)
             .index.tolist())

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="label", y=score_col, order=order,
                palette=PALETTE, ax=ax, showfliers=True)
    sns.stripplot(data=df, x="label", y=score_col, order=order,
                  color="0.3", size=3, alpha=0.4, ax=ax, jitter=True)
    ax.set_xlabel("")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, filename)
