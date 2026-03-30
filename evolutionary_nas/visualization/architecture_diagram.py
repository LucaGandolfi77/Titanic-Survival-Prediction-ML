"""
Architecture Diagram
====================
Block diagrams for best MLP and CNN architectures found by NAS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_architecture_mlp(
    config: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Best MLP Architecture",
) -> None:
    """Draw a block diagram of an MLP architecture."""
    sizes = config["hidden_sizes"]
    act = config["activation"]
    bn = config["use_batch_norm"]
    do = config["dropout_rate"]

    blocks = ["Input"] + [f"FC-{s}\n{act}" + ("\n+BN" if bn else "") +
              (f"\n+DO({do:.2f})" if do > 0 else "")
              for s in sizes] + ["Output"]

    fig, ax = plt.subplots(figsize=(max(3 * len(blocks), 8), 4))
    ax.set_xlim(-0.5, len(blocks) - 0.5)
    ax.set_ylim(-1, 2)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    colors = plt.cm.Set3(np.linspace(0, 1, len(blocks)))

    for i, (label, color) in enumerate(zip(blocks, colors)):
        rect = mpatches.FancyBboxPatch(
            (i - 0.35, 0), 0.7, 1.2, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(i, 0.6, label, ha="center", va="center", fontsize=9,
                fontweight="bold")
        if i > 0:
            ax.annotate("", xy=(i - 0.35, 0.6), xytext=(i - 1 + 0.35, 0.6),
                        arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_cnn(
    config: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Best CNN Architecture",
) -> None:
    """Draw a block diagram of a CNN architecture."""
    filters = config["filters"]
    k = config["kernel_size"]
    pool = config["pooling_type"]
    act = config["activation"]
    bn = config["use_batch_norm"]
    dw = config.get("use_depthwise", False)
    skip = config.get("use_skip_conn", False)
    dl = config["dense_layers"]
    dw_size = config["dense_width"]

    conv_labels = []
    for f in filters:
        label = f"Conv-{f}\n{k}×{k}"
        if dw:
            label = f"DWConv-{f}\n{k}×{k}"
        if bn:
            label += "\n+BN"
        label += f"\n{act}"
        if pool != "none":
            label += f"\n{pool}Pool"
        conv_labels.append(label)

    dense_labels = [f"FC-{dw_size}\n{act}" for _ in range(dl)]
    blocks = ["Input"] + conv_labels + ["GAP"] + dense_labels + ["Output"]

    fig, ax = plt.subplots(figsize=(max(2.5 * len(blocks), 10), 4.5))
    ax.set_xlim(-0.5, len(blocks) - 0.5)
    ax.set_ylim(-1, 2.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    n_conv = len(conv_labels)
    colors = []
    for i, _ in enumerate(blocks):
        if i == 0 or i == len(blocks) - 1:
            colors.append("#AEDFF7")
        elif i <= n_conv:
            colors.append("#FFD6A5")
        elif blocks[i] == "GAP":
            colors.append("#CAFFBF")
        else:
            colors.append("#FFC6FF")

    for i, (label, color) in enumerate(zip(blocks, colors)):
        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0), 0.8, 1.6, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(i, 0.8, label, ha="center", va="center", fontsize=8,
                fontweight="bold")
        if i > 0:
            ax.annotate("", xy=(i - 0.4, 0.8), xytext=(i - 1 + 0.4, 0.8),
                        arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

    if skip and n_conv > 1:
        ax.annotate("", xy=(1 - 0.2, 1.7), xytext=(n_conv - 0.2, 1.7),
                     arrowprops=dict(arrowstyle="<->", lw=1.5, color="green",
                                     connectionstyle="arc3,rad=-0.3"))
        ax.text((1 + n_conv) / 2, 2.1, "skip", ha="center", fontsize=8,
                color="green", fontstyle="italic")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    mlp_cfg = {"hidden_sizes": [256, 128, 64], "activation": "relu",
               "use_batch_norm": True, "dropout_rate": 0.2}
    plot_architecture_mlp(mlp_cfg, Path("/tmp/mlp_arch.png"))

    cnn_cfg = {"filters": [32, 64, 128], "kernel_size": 3, "use_depthwise": False,
               "use_skip_conn": True, "pooling_type": "max", "activation": "relu",
               "use_batch_norm": True, "dense_layers": 2, "dense_width": 256,
               "dropout_rate": 0.1}
    plot_architecture_cnn(cnn_cfg, Path("/tmp/cnn_arch.png"))
    print("Architecture diagrams saved.")
