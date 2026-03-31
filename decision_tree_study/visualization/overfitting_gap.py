"""
Figure: Overfitting Gap Analysis
==================================
Visualise the gap between train and test accuracy across strategies,
noise levels, or depths.  A large gap signals overfitting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.style import apply_style, save_fig, strategy_color, strategy_label


def plot_overfitting_gap_by_strategy(
    df: pd.DataFrame,
    filename: str = "overfitting_gap_strategy",
) -> Path:
    """Box plot of overfitting_gap per strategy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(df["strategy"].unique())
    palette = {strategy_label(s): strategy_color(s) for s in order}
    plot_df = df.copy()
    plot_df["Strategy"] = plot_df["strategy"].map(strategy_label)
    sns.boxplot(
        data=plot_df,
        x="Strategy",
        y="overfitting_gap",
        order=[strategy_label(s) for s in order],
        palette=palette,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Overfitting Gap (train acc − test acc)")
    ax.set_title("Overfitting Gap by Pruning Strategy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return save_fig(fig, filename)


def plot_overfitting_gap_vs_depth(
    df: pd.DataFrame,
    filename: str = "overfitting_gap_depth",
) -> Path:
    """Line plot of mean overfitting gap vs max_depth."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby("max_depth")["overfitting_gap"].agg(["mean", "std"]).reset_index()
    ax.plot(grouped["max_depth"], grouped["mean"], "-o", linewidth=2)
    ax.fill_between(
        grouped["max_depth"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.15,
    )
    ax.set_xlabel("max_depth (-1 = unlimited)")
    ax.set_ylabel("Overfitting Gap")
    ax.set_title("Overfitting Gap vs Max Depth")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    return save_fig(fig, filename)
