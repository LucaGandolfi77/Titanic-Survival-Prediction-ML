"""pipeline.py — Reusable functions for Iris classification lab."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Data loading & splitting
# ------------------------------------------------------------------

def load_and_split(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Load Iris and split into train / validation / test sets.

    Returns a dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_names, target_names
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    # First split: separate out the test set
    # val_size is relative to the full dataset, so the second split
    # needs the fraction relative to the remaining data.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=random_state,
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": iris.feature_names,
        "target_names": list(iris.target_names),
    }


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              k: int = 5) -> KNeighborsClassifier:
    """Fit a kNN classifier with the given k."""
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return clf


def select_best_knn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    k_values: list[int] | None = None,
) -> tuple[KNeighborsClassifier, dict[int, float]]:
    """Try several k values, pick the one with highest validation accuracy.

    Returns (best_model, {k: val_accuracy}).
    """
    if k_values is None:
        k_values = [3, 5, 7]

    results: dict[int, float] = {}
    best_acc = -1.0
    best_model: KNeighborsClassifier | None = None

    for k in k_values:
        model = train_knn(X_train, y_train, k=k)
        acc = accuracy_score(y_val, model.predict(X_val))
        results[k] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model, results  # type: ignore[return-value]


def train_decision_tree(
    X_train: np.ndarray, y_train: np.ndarray,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Fit a Decision Tree classifier."""
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    set_label: str = "Test",
) -> tuple[float, str, np.ndarray]:
    """Compute accuracy, classification report, and confusion matrix.

    Returns (accuracy, report_str, cm_array).
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=target_names)
    cm = confusion_matrix(y, y_pred)
    return acc, report, cm


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_confusion_matrices(
    cm_knn: np.ndarray,
    cm_dt: np.ndarray,
    target_names: list[str],
    save_path: str = "outputs/confusion_matrices.png",
) -> None:
    """Plot two confusion matrices side by side and save to file."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(cm_knn, display_labels=target_names).plot(
        ax=axes[0], cmap="Blues", colorbar=False,
    )
    axes[0].set_title("kNN")

    ConfusionMatrixDisplay(cm_dt, display_labels=target_names).plot(
        ax=axes[1], cmap="Greens", colorbar=False,
    )
    axes[1].set_title("Decision Tree")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {save_path}")


def plot_accuracy_comparison(
    knn_val_acc: float, knn_test_acc: float,
    dt_val_acc: float, dt_test_acc: float,
    save_path: str = "outputs/accuracy_comparison.png",
) -> None:
    """Bar chart comparing validation and test accuracy for both models."""
    labels = ["kNN", "Decision Tree"]
    val_accs = [knn_val_acc, dt_val_acc]
    test_accs = [knn_test_acc, dt_test_acc]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_val = ax.bar(x - width / 2, val_accs, width, label="Validation")
    bars_test = ax.bar(x + width / 2, test_accs, width, label="Test")

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Annotate bars
    for bar in (*bars_val, *bars_test):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {save_path}")
