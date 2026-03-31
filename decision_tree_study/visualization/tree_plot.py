"""
Figure: Graphviz-style decision tree visualisation
====================================================
Render a fitted tree as a matplotlib figure using sklearn's plot_tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from visualization.style import apply_style, save_fig


def plot_decision_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str] | None = None,
    class_names: List[str] | None = None,
    title: str = "Decision Tree",
    filename: str = "tree_plot",
    max_depth: int = 4,
) -> Path:
    """Render a decision tree and save it."""
    apply_style()
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        max_depth=max_depth,
        fontsize=8,
    )
    ax.set_title(title)
    return save_fig(fig, filename)
