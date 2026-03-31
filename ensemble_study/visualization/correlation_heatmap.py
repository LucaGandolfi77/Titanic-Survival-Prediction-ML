"""Spearman ρ correlation heatmap between all numeric experiment columns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization._common import _save


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Spearman Rank-Correlation Matrix",
    filename: str = "correlation_heatmap",
) -> Path:
    if columns:
        numeric = df[columns]
    else:
        numeric = df.select_dtypes(include=[np.number])

    corr = numeric.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, filename)
