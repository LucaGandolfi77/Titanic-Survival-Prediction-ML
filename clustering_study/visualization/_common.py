"""Shared helpers for all visualisation modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from config import CFG
from algorithms.algorithm_factory import METHOD_LABELS

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("colorblind", n_colors=len(CFG.METHOD_NAMES))
METHOD_COLORS = {m: PALETTE[i] for i, m in enumerate(CFG.METHOD_NAMES)}


def _save(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    d = Path(CFG.PLOTS_DIR)
    if subdir:
        d = d / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path_pdf = d / f"{name}.pdf"
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return path


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)
