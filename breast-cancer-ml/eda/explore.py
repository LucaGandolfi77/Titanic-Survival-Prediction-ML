"""Exploratory Data Analysis for the Breast Cancer dataset.

Generates:
    - Class distribution bar chart
    - Correlation heatmap (top 15 features)
    - Box plots of the top 5 most discriminative features

All figures are saved as PNG under ``outputs/plots/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402

from data.loader import ROOT, load_config, get_feature_importances  # noqa: E402


# ── styling defaults ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
DPI = 150


def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────
#  1. Class distribution
# ─────────────────────────────────────────────────────────────────
def plot_class_distribution(
    y: pd.Series,
    target_names: list[str],
    save_dir: Path,
) -> None:
    """Bar chart showing class balance.

    Args:
        y: Full target series (before split).
        target_names: Human-readable class labels.
        save_dir: Directory to save the PNG.
    """
    counts = y.value_counts().sort_index()
    labels = [target_names[i] for i in counts.index]
    colours = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colours, edgecolor="white", width=0.55)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    ax.set_title("Class Distribution", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_dir / "class_distribution.png", dpi=DPI)
    plt.close(fig)
    print("  ✓ Saved class_distribution.png")


# ─────────────────────────────────────────────────────────────────
#  2. Correlation heatmap (top 15 features)
# ─────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Path,
    top_n: int = 15,
) -> None:
    """Heatmap of pairwise correlations among the *top_n* features.

    Features are ranked by their absolute correlation with the target.

    Args:
        X: Feature matrix.
        y: Target vector.
        save_dir: Directory to save the PNG.
        top_n: Number of features to include.
    """
    importance = get_feature_importances(X, y)
    top_features = importance.head(top_n).index.tolist()

    corr = X[top_features].corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap — Top {top_n} Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "correlation_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  ✓ Saved correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────
#  3. Box plots of top 5 discriminative features
# ─────────────────────────────────────────────────────────────────
def plot_top_features_boxplots(
    X: pd.DataFrame,
    y: pd.Series,
    target_names: list[str],
    save_dir: Path,
    top_n: int = 5,
) -> None:
    """Box plots comparing class distributions for the *top_n* features.

    Args:
        X: Feature matrix.
        y: Target vector.
        target_names: Human-readable class labels.
        save_dir: Directory to save the PNG.
        top_n: Number of features to plot.
    """
    importance = get_feature_importances(X, y)
    top_features = importance.head(top_n).index.tolist()

    plot_df = X[top_features].copy()
    plot_df["class"] = y.map(lambda v: target_names[v])

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 5), sharey=False)
    if top_n == 1:
        axes = [axes]

    palette = {"malignant": "#e74c3c", "benign": "#2ecc71"}
    for ax, feat in zip(axes, top_features):
        sns.boxplot(
            data=plot_df,
            x="class",
            y=feat,
            palette=palette,
            ax=ax,
            width=0.5,
            fliersize=3,
        )
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Top {top_n} Discriminative Features", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "top_features_boxplots.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved top_features_boxplots.png")


# ─────────────────────────────────────────────────────────────────
#  4. Feature-importance bar chart
# ─────────────────────────────────────────────────────────────────
def plot_feature_importance_bar(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of absolute target correlation.

    Args:
        X: Feature matrix.
        y: Target vector.
        save_dir: Directory to save the PNG.
        top_n: Number of features to show.
    """
    importance = get_feature_importances(X, y).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = sns.color_palette("viridis", n_colors=top_n)
    ax.barh(importance.index[::-1], importance.values[::-1], color=colours[::-1], edgecolor="white")
    ax.set_xlabel("|Correlation with target|")
    ax.set_title(f"Top {top_n} Features by Target Correlation", fontsize=14, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_dir / "feature_importance.png", dpi=DPI)
    plt.close(fig)
    print("  ✓ Saved feature_importance.png")


# ─────────────────────────────────────────────────────────────────
#  Run all
# ─────────────────────────────────────────────────────────────────
def run_eda(config: Optional[dict] = None) -> None:
    """Execute the full EDA pipeline.

    Args:
        config: Configuration dict.  Loaded from disk when *None*.
    """
    if config is None:
        config = load_config()

    plots_dir = _ensure_dir(ROOT / config["paths"]["plots"])

    # Load raw (unscaled) data for EDA
    bunch = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = bunch.data  # type: ignore[assignment]
    y: pd.Series = bunch.target  # type: ignore[assignment]
    target_names: list[str] = list(bunch.target_names)

    print("── Exploratory Data Analysis ─────────────────────")
    plot_class_distribution(y, target_names, plots_dir)
    plot_correlation_heatmap(X, y, plots_dir)
    plot_top_features_boxplots(X, y, target_names, plots_dir)
    plot_feature_importance_bar(X, y, plots_dir)
    print("── Done ──────────────────────────────────────────\n")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()
