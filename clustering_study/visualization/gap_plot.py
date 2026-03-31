"""Gap statistic plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _save


def plot_gap_statistic(
    k_range: list,
    gaps: list,
    sk: list,
    best_k: int | None = None,
    title: str = "Gap Statistic",
    filename: str = "gap_statistic",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(k_range, gaps, yerr=sk, marker="o", capsize=4,
                color="#1f77b4", linewidth=2)
    if best_k is not None:
        ax.axvline(best_k, linestyle="--", color="#d62728", alpha=0.7,
                   label=f"Best k={best_k}")
        ax.legend()
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Gap(k)")
    ax.set_title(title)
    ax.set_xticks(k_range)
    return _save(fig, filename, subdir="k_selection")
