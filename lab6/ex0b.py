#!/usr/bin/env python3
"""
Exercise 0b — GP-based Image Denoising & Enhancement
=====================================================

**Part A  – Denoising**
  Load/generate a grayscale image I, create noisy O = clip(I + U(-15,15)),
  evolve a GP filter  f(3×3 neighbourhood of O) ≈ I.
  Test generalisation on a *different* image (and/or noise level).

**Part B  – Enhancement**
  Load a low-quality image I, produce an enhanced target O (contrast +
  sharpening), then evolve a GP tree that maps I → O per-pixel using a
  3×3 neighbourhood.  Apply the tree to a *different* unseen image.

**Terminals** (for both parts):
  9 neighbourhood pixels  p0..p8  +  Ephemeral Random Constants in [0, K].

**Function set** (all outputs clipped to [0, 255]):
  clipped_add, clipped_sub, clipped_mul, clipped_div, clipped_avg,
  clipped_max, clipped_min, clipped_abs, edge_detect, local_contrast.

Custom operators documented in utils.py:
  - edge_detect(a, b) = clip(|a - b|)  — simple edge detector
  - local_contrast(a, b) = clip(128 + 1.5*(a - b)) — local contrast boost
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Callable, List, Tuple

import numpy as np
from deap import algorithms, base, creator, gp, tools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    RESULTS_DIR, IMAGES_DIR,
    clipped_add, clipped_sub, clipped_mul, clipped_div,
    clipped_avg, clipped_max, clipped_min, clipped_abs,
    edge_detect, local_contrast,
    load_grayscale, save_grayscale, add_uniform_noise,
    generate_synthetic_image, generate_enhanced_image,
    mse, psnr, ssim,
    plot_images_row, plot_convergence,
)

# ===================== HYPERPARAMETERS ====================================
POP_SIZE       = 300
NGEN           = 40
CXPB           = 0.7
MUTPB          = 0.2
TOURN_SIZE     = 5
MAX_DEPTH_INIT = 3
MAX_DEPTH      = 8        # bloat control
ERC_MAX        = 10.0     # K — ephemeral random constant upper bound
NOISE_LEVEL    = 15
SEED           = 42
# Subsample: evaluate GP on every Kth pixel to keep runtime manageable
EVAL_STEP      = 4
# ==========================================================================


# ----------------------- primitive set ------------------------------------

def _make_pset() -> gp.PrimitiveSet:
    """Build the DEAP primitive set for 3×3 neighbourhood inputs.

    Terminals: p0 … p8 (9 pixels) + ERC ∈ [0, K].
    """
    pset = gp.PrimitiveSet("IMG", arity=9)
    # Rename arguments to meaningful names
    names = {f"ARG{i}": f"p{i}" for i in range(9)}
    pset.renameArguments(**names)

    # Function set — all clipped to [0, 255]
    pset.addPrimitive(clipped_add, 2, name="add")
    pset.addPrimitive(clipped_sub, 2, name="sub")
    pset.addPrimitive(clipped_mul, 2, name="mul")
    pset.addPrimitive(clipped_div, 2, name="div")
    pset.addPrimitive(clipped_avg, 2, name="avg")
    pset.addPrimitive(clipped_max, 2, name="mx")
    pset.addPrimitive(clipped_min, 2, name="mn")
    pset.addPrimitive(clipped_abs, 1, name="ab")
    pset.addPrimitive(edge_detect,   2, name="edge")
    pset.addPrimitive(local_contrast, 2, name="lcon")

    pset.addEphemeralConstant("ERC", lambda: random.uniform(0.0, ERC_MAX))
    return pset


# ----------------------- neighbourhood extraction -------------------------

def _get_neighbours(img: np.ndarray, r: int, c: int) -> Tuple[float, ...]:
    """Extract the 3×3 neighbourhood of (r, c) with replicate padding.

    Order: top-left, top, top-right, left, center, right,
           bottom-left, bottom, bottom-right  →  p0 … p8.

    Args:
        img: 2-D float64 array.
        r, c: pixel coordinates.

    Returns:
        Tuple of 9 float values.
    """
    h, w = img.shape
    vals: list[float] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = max(0, min(h - 1, r + dr))
            cc = max(0, min(w - 1, c + dc))
            vals.append(float(img[rr, cc]))
    return tuple(vals)


# ----------------------- GP evaluation ------------------------------------

def _eval_filter(individual, toolbox, source: np.ndarray, target: np.ndarray,
                 step: int = EVAL_STEP) -> Tuple[float]:
    """MSE between GP-filtered source and target, sub-sampled for speed.

    Args:
        individual: GP tree.
        toolbox: must have ``compile``.
        source: input (noisy / low-quality) image.
        target: ground-truth / enhanced image.
        step: evaluate every *step*-th pixel.

    Returns:
        (mse,)
    """
    func = toolbox.compile(expr=individual)
    h, w = source.shape
    total = 0.0
    count = 0
    for r in range(1, h - 1, step):
        for c in range(1, w - 1, step):
            nb = _get_neighbours(source, r, c)
            try:
                pred = func(*nb)
                if not np.isfinite(pred):
                    pred = 128.0
            except (OverflowError, ValueError, ZeroDivisionError):
                pred = 128.0
            pred = float(np.clip(pred, 0.0, 255.0))
            total += (pred - target[r, c]) ** 2
            count += 1
    return (total / max(count, 1),)


def apply_filter(func: Callable, source: np.ndarray) -> np.ndarray:
    """Apply a compiled GP tree to every pixel of an image.

    Args:
        func: compiled GP callable accepting 9 floats.
        source: input image (H×W float64).

    Returns:
        Filtered image (H×W float64, clipped to [0,255]).
    """
    h, w = source.shape
    out = np.copy(source)
    for r in range(h):
        for c in range(w):
            nb = _get_neighbours(source, r, c)
            try:
                val = func(*nb)
                if not np.isfinite(val):
                    val = 128.0
            except (OverflowError, ValueError, ZeroDivisionError):
                val = 128.0
            out[r, c] = float(np.clip(val, 0.0, 255.0))
    return out


# ----------------------- core GP runner -----------------------------------

def run_image_gp(
    source: np.ndarray,
    target: np.ndarray,
    pop_size: int = POP_SIZE,
    ngen: int = NGEN,
    cxpb: float = CXPB,
    mutpb: float = MUTPB,
    seed: int = SEED,
    verbose: bool = True,
    label: str = "ImageGP",
) -> Tuple[object, tools.Logbook]:
    """Evolve a GP filter mapping *source* → *target*.

    Returns:
        (best_individual, logbook)
    """
    random.seed(seed)
    np.random.seed(seed)

    pset = _make_pset()

    for name in ("FitnessMin", "Individual"):
        if name in creator.__dict__:
            del creator.__dict__[name]
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", _eval_filter, toolbox=toolbox,
                     source=source, target=target)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=lambda i: i.height, max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda i: i.height, max_value=MAX_DEPTH))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    if verbose:
        print(f"\n  [{label}] pop={pop_size}, ngen={ngen}, cx={cxpb}, mut={mutpb}")

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)
    return hof[0], logbook


# ===================== PART A  – Denoising ================================

def part_a() -> None:
    """Run GP-based denoising and generalization test."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("=" * 65)
    print("  Exercise 0b — Part A: GP Image Denoising")
    print("=" * 65)

    # --- images ---
    train_img = generate_synthetic_image(128, 128, seed=0)
    test_img  = generate_synthetic_image(128, 128, seed=99)
    save_grayscale(train_img, os.path.join(IMAGES_DIR, "train_clean.png"))
    save_grayscale(test_img,  os.path.join(IMAGES_DIR, "test_clean.png"))

    noisy_train = add_uniform_noise(train_img, level=NOISE_LEVEL, seed=10)
    noisy_test  = add_uniform_noise(test_img,  level=NOISE_LEVEL, seed=20)
    save_grayscale(noisy_train, os.path.join(IMAGES_DIR, "train_noisy.png"))
    save_grayscale(noisy_test,  os.path.join(IMAGES_DIR, "test_noisy.png"))

    # --- evolve filter on training image ---
    best, logbook = run_image_gp(noisy_train, train_img, label="Denoise-Train", verbose=True)
    print(f"\n  Best tree (denoising): {best}")
    print(f"  Best training MSE:     {best.fitness.values[0]:.4f}")

    # --- apply to both images ---
    pset = _make_pset()
    func = gp.compile(best, pset)

    filtered_train = apply_filter(func, noisy_train)
    filtered_test  = apply_filter(func, noisy_test)

    # --- metrics ---
    for tag, orig, noisy, filt in [
        ("Train", train_img, noisy_train, filtered_train),
        ("Test",  test_img,  noisy_test,  filtered_test),
    ]:
        p_noisy = psnr(orig, noisy)
        p_filt  = psnr(orig, filt)
        s_noisy = ssim(orig, noisy)
        s_filt  = ssim(orig, filt)
        print(f"  {tag}:  PSNR noisy={p_noisy:.2f} dB → filtered={p_filt:.2f} dB  |"
              f"  SSIM noisy={s_noisy:.4f} → filtered={s_filt:.4f}")

    # --- save visualisations ---
    plot_images_row(
        [train_img, noisy_train, filtered_train],
        ["Clean (train)", f"Noisy (±{NOISE_LEVEL})", "GP Denoised"],
        os.path.join(RESULTS_DIR, "ex0b_denoise_train.png"),
    )
    plot_images_row(
        [test_img, noisy_test, filtered_test],
        ["Clean (test)", f"Noisy (±{NOISE_LEVEL})", "GP Denoised (generalization)"],
        os.path.join(RESULTS_DIR, "ex0b_denoise_test.png"),
    )
    plot_convergence(
        {"Best MSE": logbook.select("min"), "Avg MSE": logbook.select("avg")},
        "Ex 0b Part A — Denoising Convergence",
        os.path.join(RESULTS_DIR, "ex0b_denoise_convergence.png"),
        ylabel="MSE",
    )


# ===================== PART B  – Enhancement ==============================

def part_b() -> None:
    """Run GP-based image enhancement and generalization test."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Exercise 0b — Part B: GP Image Enhancement")
    print("=" * 65)

    # --- images ---
    train_raw = generate_synthetic_image(128, 128, seed=5)
    test_raw  = generate_synthetic_image(128, 128, seed=77)
    train_enhanced = generate_enhanced_image(train_raw)
    test_enhanced  = generate_enhanced_image(test_raw)

    save_grayscale(train_raw,      os.path.join(IMAGES_DIR, "enhance_train_raw.png"))
    save_grayscale(train_enhanced,  os.path.join(IMAGES_DIR, "enhance_train_target.png"))
    save_grayscale(test_raw,       os.path.join(IMAGES_DIR, "enhance_test_raw.png"))
    save_grayscale(test_enhanced,   os.path.join(IMAGES_DIR, "enhance_test_target.png"))

    # --- evolve enhancement tree ---
    best, logbook = run_image_gp(train_raw, train_enhanced,
                                  label="Enhance-Train", verbose=True)
    print(f"\n  Best tree (enhancement): {best}")
    print(f"  Best training MSE:       {best.fitness.values[0]:.4f}")

    pset = _make_pset()
    func = gp.compile(best, pset)

    enhanced_train = apply_filter(func, train_raw)
    enhanced_test  = apply_filter(func, test_raw)

    for tag, target, result in [
        ("Train", train_enhanced, enhanced_train),
        ("Test",  test_enhanced,  enhanced_test),
    ]:
        p = psnr(target, result)
        s = ssim(target, result)
        print(f"  {tag}:  PSNR={p:.2f} dB  |  SSIM={s:.4f}")

    plot_images_row(
        [train_raw, train_enhanced, enhanced_train],
        ["Raw (train)", "Target Enhanced", "GP Enhanced"],
        os.path.join(RESULTS_DIR, "ex0b_enhance_train.png"),
    )
    plot_images_row(
        [test_raw, test_enhanced, enhanced_test],
        ["Raw (test)", "Target Enhanced", "GP Enhanced (generalization)"],
        os.path.join(RESULTS_DIR, "ex0b_enhance_test.png"),
    )
    plot_convergence(
        {"Best MSE": logbook.select("min"), "Avg MSE": logbook.select("avg")},
        "Ex 0b Part B — Enhancement Convergence",
        os.path.join(RESULTS_DIR, "ex0b_enhance_convergence.png"),
        ylabel="MSE",
    )


# ===================== main ===============================================

def main() -> None:
    """Run both parts of Exercise 0b."""
    part_a()
    part_b()
    print("\n  All done — plots in results/, images in images/")


if __name__ == "__main__":
    main()
