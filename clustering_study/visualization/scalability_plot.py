"""Log-log scalability plot (fit time vs data size)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import METHOD_COLORS, _save, method_label


def plot_scalability(
    df: pd.DataFrame,
    filename: str = "scalability",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m].groupby("n_samples")["fit_time_ms"].agg(
            ["mean", "std"]
        )
        colour = METHOD_COLORS.get(m, None)
        ax.errorbar(
            sub.index,
            sub["mean"],
            yerr=sub["std"],
            fmt="o-",
            label=method_label(m),
            color=colour,
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Fit Time (ms)")
    ax.set_title("Scalability — Fit Time vs Data Size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, filename, subdir="scalability")
