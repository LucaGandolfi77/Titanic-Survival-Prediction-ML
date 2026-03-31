"""
Shared plotting helpers
========================
Consistent styling, colour palette, save logic, and axis formatters
used by every figure module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import CFG, ensure_dirs

# ── Global style ──────────────────────────────────────────────────────

PALETTE = sns.color_palette("colorblind")
STRATEGY_COLORS = {
    "none": PALETTE[0],
    "pre_depth": PALETTE[1],
    "pre_samples": PALETTE[2],
    "ccp": PALETTE[3],
    "combined": PALETTE[4],
}
STRATEGY_LABELS = {
    "none": "No pruning",
    "pre_depth": "Pre-pruning (depth)",
    "pre_samples": "Pre-pruning (samples)",
    "ccp": "CCP",
    "combined": "Combined",
}

FIG_DPI = 300
FIG_FORMAT = "png"


def apply_style() -> None:
    """Set the global matplotlib / seaborn style."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def save_fig(fig: plt.Figure, name: str, subdir: str = "plots") -> Path:
    """Save a figure to the results directory and close it."""
    ensure_dirs()
    out = Path(CFG.RESULTS_DIR) / subdir / f"{name}.{FIG_FORMAT}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def strategy_color(s: str) -> tuple:
    return STRATEGY_COLORS.get(s, PALETTE[5])


def strategy_label(s: str) -> str:
    return STRATEGY_LABELS.get(s, s)
