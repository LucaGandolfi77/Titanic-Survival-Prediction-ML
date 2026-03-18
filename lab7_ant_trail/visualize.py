"""visualize.py — Plotting utilities for the Ant Trail experiment.

Produces grouped bar charts comparing model scores across
configurations (m, num_training_games, model_type).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


# -----------------------------------------------------------------------
# Grouped bar chart
# -----------------------------------------------------------------------

def plot_grouped_bar(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Grouped bar chart: x = (m, num_games), bars = model_type, y = mean score.

    Args:
        df: results DataFrame with columns [m, num_training_games,
            model_type, test_board_id, score].
        save_path: if given, save the figure to this path.
    """
    summary = (df.groupby(["m", "num_training_games", "model_type"])["score"]
               .mean().reset_index())

    configs = summary.groupby(["m", "num_training_games"]).ngroup()
    config_labels = (summary.groupby(["m", "num_training_games"])
                     .first().index
                     .map(lambda t: f"m={t[0]}, g={t[1]}"))
    model_types = sorted(summary["model_type"].unique())
    n_models = len(model_types)
    n_configs = len(config_labels)

    x = np.arange(n_configs)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mt in enumerate(model_types):
        sub = summary[summary["model_type"] == mt].sort_values(
            ["m", "num_training_games"])
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, sub["score"].values, width, label=mt.upper())

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean Test Score")
    ax.set_title("Ant Trail — Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Figure saved to {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Per-model box plot
# -----------------------------------------------------------------------

def plot_boxplot(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Box plot of score distributions per model type.

    Args:
        df: results DataFrame.
        save_path: optional save path.
    """
    model_types = sorted(df["model_type"].unique())
    data = [df[df["model_type"] == mt]["score"].values for mt in model_types]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[mt.upper() for mt in model_types])
    ax.set_ylabel("Test Score")
    ax.set_title("Score Distribution by Model Type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Figure saved to {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# All-in-one
# -----------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame | None = None) -> None:
    """Load results (or use provided DataFrame) and produce all plots."""
    if df is None:
        path = os.path.join(RESULTS_DIR, "results.csv")
        if not os.path.exists(path):
            print(f"  No results file found at {path}")
            return
        df = pd.read_csv(path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_grouped_bar(df, os.path.join(RESULTS_DIR, "grouped_bar.png"))
    plot_boxplot(df, os.path.join(RESULTS_DIR, "boxplot.png"))
    print("  All plots generated.")


# -----------------------------------------------------------------------
# Standalone
# -----------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_plots()
