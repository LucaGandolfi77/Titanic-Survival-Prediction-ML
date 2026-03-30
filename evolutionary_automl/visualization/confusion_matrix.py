"""
Confusion matrix visualization for the best discovered pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix — Best Pipeline",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix for a fitted pipeline on test data.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Test labels.
        class_names: Optional list of class names.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))])
    pipe.fit(X_train, y_train)
    plot_confusion_matrix(pipe, X_test, y_test, class_names=["setosa", "versicolor", "virginica"])
    plt.show()
