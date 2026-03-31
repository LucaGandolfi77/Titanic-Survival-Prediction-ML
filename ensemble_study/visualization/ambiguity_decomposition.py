"""Ambiguity decomposition — stacked bar chart (ensemble error, ambiguity)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _save, method_label


def plot_ambiguity_decomposition(
    df: pd.DataFrame,
    method_col: str = "method",
    title: str = "Ambiguity Decomposition (Krogh–Vedelsby)",
    filename: str = "ambiguity_decomposition",
) -> Path:
    """Stacked bar: avg_individual_error = ensemble_error + ambiguity."""
    agg = df.groupby(method_col).agg(
        ensemble_error=("ensemble_error", "mean"),
        ambiguity=("ambiguity", "mean"),
        avg_individual_error=("avg_individual_error", "mean"),
    ).reset_index()
    agg = agg.sort_values("avg_individual_error", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [method_label(m) for m in agg[method_col]]
    x = np.arange(len(labels))
    w = 0.6

    ax.bar(x, agg["ensemble_error"], w, label="Ensemble Error", color="#d62728")
    ax.bar(x, agg["ambiguity"], w, bottom=agg["ensemble_error"],
           label="Ambiguity (diversity bonus)", color="#2ca02c", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, filename, subdir="exp5")
