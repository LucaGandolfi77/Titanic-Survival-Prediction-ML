"""
Figure: Feature Importance bar chart
======================================
Horizontal bar chart of feature importances for a fitted tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from visualization.style import apply_style, save_fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str] | None = None,
    title: str = "Feature Importance",
    filename: str = "feature_importance",
    top_k: int = 20,
) -> Path:
    """Horizontal bar chart of top-k feature importances."""
    apply_style()
    n = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]

    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(range(len(names)), values, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, filename)
