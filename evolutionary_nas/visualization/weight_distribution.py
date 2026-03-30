"""
Weight Distribution
===================
Histogram of weight values for the best trained model.
Also includes learning curve plot for the best model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


def plot_weight_distribution(
    model: nn.Module,
    save_path: Optional[Path] = None,
    title: str = "Weight Distribution of Best Model",
) -> None:
    """Histogram of all weight parameter values in the model."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    all_weights: List[float] = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.extend(param.detach().cpu().numpy().flatten().tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_weights, bins=100, edgecolor="black", linewidth=0.3,
            alpha=0.8, color="C0")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    mean_w = np.mean(all_weights)
    std_w = np.std(all_weights)
    ax.text(0.72, 0.85, f"μ={mean_w:.4f}\nσ={std_w:.4f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[Path] = None,
    title: str = "Learning Curve — Best Model",
) -> None:
    """Plot train/val loss and accuracy curves."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_accs, label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    plot_weight_distribution(model, Path("/tmp/weight_hist.png"))
    plot_learning_curve(
        [1.0, 0.8, 0.6, 0.5, 0.4], [1.1, 0.9, 0.7, 0.65, 0.55],
        [0.5, 0.6, 0.7, 0.75, 0.8], [0.45, 0.55, 0.65, 0.7, 0.75],
        Path("/tmp/learning_curve.png"),
    )
    print("Plots saved.")
