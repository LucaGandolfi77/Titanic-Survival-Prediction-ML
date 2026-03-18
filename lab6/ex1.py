#!/usr/bin/env python3
"""
Exercise 1 (BONUS) — GP Image Enhancement Generalization Study
==============================================================

Repeats Exercise 0b Part B with a *systematic* generalization study:

  • Use at least 2 different source/target image pairs for training.
  • For each trained GP tree, evaluate on ALL pairs (including unseen ones).
  • Report PSNR and SSIM for every (train-pair, eval-pair) combination.
  • Plot a grouped bar-chart summarising the metrics.

This exercise reuses the GP machinery from ex0b via the shared helpers.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
from deap import gp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    RESULTS_DIR, IMAGES_DIR,
    generate_synthetic_image, generate_enhanced_image,
    save_grayscale, psnr, ssim,
    plot_images_row,
)
from ex0b import run_image_gp, apply_filter, _make_pset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== HYPERPARAMETERS ====================================
POP_SIZE    = 300
NGEN        = 40
CXPB        = 0.7
MUTPB       = 0.2
SEED        = 42
# ==========================================================================

# ----------------------- image pairs --------------------------------------

def _build_pairs() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate 3 (raw, enhanced) image pairs with different structures.

    Returns:
        List of (label, raw_image, enhanced_image).
    """
    pairs = []
    for idx, seed in enumerate([5, 77, 123]):
        raw = generate_synthetic_image(128, 128, seed=seed)
        enh = generate_enhanced_image(raw)
        pairs.append((f"Pair-{idx}", raw, enh))
    return pairs


# ===================== main ===============================================

def main() -> None:
    """Train on each pair, evaluate on all pairs, report metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("=" * 65)
    print("  Exercise 1 (BONUS) — GP Enhancement Generalization Study")
    print("=" * 65)

    pairs = _build_pairs()

    # Save all pair images
    for label, raw, enh in pairs:
        save_grayscale(raw, os.path.join(IMAGES_DIR, f"ex1_{label}_raw.png"))
        save_grayscale(enh, os.path.join(IMAGES_DIR, f"ex1_{label}_enhanced.png"))

    # Train one GP tree per pair
    trained_trees: List[Tuple[str, object]] = []
    for label, raw, enh in pairs:
        print(f"\n>>> Training on {label} …")
        best, _ = run_image_gp(raw, enh, pop_size=POP_SIZE, ngen=NGEN,
                                cxpb=CXPB, mutpb=MUTPB,
                                seed=SEED, verbose=False, label=label)
        print(f"    Best MSE (train): {best.fitness.values[0]:.4f}")
        trained_trees.append((label, best))

    # Evaluate every tree on every pair
    pset = _make_pset()
    results: Dict[str, Dict[str, Tuple[float, float]]] = {}  # train_label → {eval_label → (psnr, ssim)}

    print("\n" + "-" * 65)
    print(f"{'Trained on':<12} {'Evaluated on':<14} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("-" * 65)

    for train_label, tree in trained_trees:
        func = gp.compile(tree, pset)
        results[train_label] = {}
        for eval_label, raw, enh in pairs:
            gp_result = apply_filter(func, raw)
            p = psnr(enh, gp_result)
            s = ssim(enh, gp_result)
            results[train_label][eval_label] = (p, s)
            marker = " ← (self)" if train_label == eval_label else ""
            print(f"  {train_label:<12} {eval_label:<14} {p:>10.2f} {s:>8.4f}{marker}")

            # Save generalisation images
            plot_images_row(
                [raw, enh, gp_result],
                [f"Raw ({eval_label})", "Target", f"GP ({train_label})"],
                os.path.join(RESULTS_DIR, f"ex1_{train_label}_on_{eval_label}.png"),
            )

    # ---- grouped bar chart --------------------------------------------------
    train_labels = [t for t, _ in trained_trees]
    eval_labels  = [l for l, _, _ in pairs]
    n_train = len(train_labels)
    n_eval  = len(eval_labels)

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(n_eval)
    width = 0.8 / n_train

    for i, tl in enumerate(train_labels):
        psnr_vals = [results[tl][el][0] for el in eval_labels]
        ssim_vals = [results[tl][el][1] for el in eval_labels]
        offset = (i - n_train / 2 + 0.5) * width
        ax_p.bar(x + offset, psnr_vals, width, label=f"Trained: {tl}")
        ax_s.bar(x + offset, ssim_vals, width, label=f"Trained: {tl}")

    for ax, metric in [(ax_p, "PSNR (dB)"), (ax_s, "SSIM")]:
        ax.set_xticks(x)
        ax.set_xticklabels(eval_labels)
        ax.set_ylabel(metric)
        ax.set_xlabel("Evaluated on")
        ax.set_title(f"{metric} — Generalization")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "ex1_generalization_metrics.png")
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\n  [plot] Saved: {chart_path}")
    print("\n  Done — all plots saved to results/")


if __name__ == "__main__":
    main()
