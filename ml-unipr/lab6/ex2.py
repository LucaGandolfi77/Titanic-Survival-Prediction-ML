#!/usr/bin/env python3
"""
Exercise 2 — PSO Function Maximisation
=======================================

**Target function:**
  f(x,y,z) = [1 + cos(2π·(1 + exp(-(x-1)²/25 - (y-1.27)²)))] / (1 + z²)

**Search space:** x, y, z ∈ [-128, 128]  TOROIDAL  (wraps around).
**Speed limit:**  |v_i| ≤ 64   with two strategies: hard-clip and reflection.

**Plan:**
  Part 1 — 50 particles, grid-search (c1, c2, w) → find best triple.
  Part 2 — 100 particles with that best triple.
  Part 3 — 100 particles, further (c1, c2, w) search.
  Topology comparison: global-best vs ring neighbourhood.
  Speed-limit comparison: hard-clip vs reflection.

All convergence curves and the final 2-D slice plot are saved as PNG.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RESULTS_DIR, plot_convergence, plot_swarm_on_surface

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== HYPERPARAMETERS ====================================
MAX_ITER   = 10_000
LO, HI     = -128.0, 128.0
V_MAX       = 64.0
DIM         = 3
SEED        = 42
# ==========================================================================


# ----------------------- target function ----------------------------------

def f(x: float, y: float, z: float) -> float:
    """Target function to MAXIMISE.

    f(x,y,z) = [1 + cos(2π·(1 + exp(-(x-1)²/25 - (y-1.27)²)))] / (1 + z²)
    """
    inner = 1.0 + math.exp(-((x - 1.0) ** 2) / 25.0 - (y - 1.27) ** 2)
    return (1.0 + math.cos(2.0 * math.pi * inner)) / (1.0 + z * z)


def f_vec(X: np.ndarray, Y: np.ndarray, z: float) -> np.ndarray:
    """Vectorised version for 2-D slice plotting (z held constant)."""
    inner = 1.0 + np.exp(-((X - 1.0) ** 2) / 25.0 - (Y - 1.27) ** 2)
    return (1.0 + np.cos(2.0 * np.pi * inner)) / (1.0 + z * z)


# ----------------------- toroidal wrap ------------------------------------

def wrap(val: float) -> float:
    """Toroidal wrap to [-128, 128)."""
    span = HI - LO  # 256
    v = val - LO
    v = v % span
    return v + LO


# ----------------------- speed limiting -----------------------------------

def clip_speed(v: np.ndarray) -> np.ndarray:
    """Hard-clip each component to [-V_MAX, V_MAX]."""
    return np.clip(v, -V_MAX, V_MAX)


def reflect_speed(v: np.ndarray) -> np.ndarray:
    """Reflection strategy: if |v_i| > V_MAX, reflect it back."""
    out = v.copy()
    for i in range(len(out)):
        while abs(out[i]) > V_MAX:
            if out[i] > V_MAX:
                out[i] = 2 * V_MAX - out[i]
            elif out[i] < -V_MAX:
                out[i] = -2 * V_MAX - out[i]
    return out


# ----------------------- PSO core -----------------------------------------

class Particle:
    """A single PSO particle with position, velocity, and personal best."""

    __slots__ = ("pos", "vel", "fitness", "best_pos", "best_fit")

    def __init__(self, rng: np.random.RandomState) -> None:
        self.pos: np.ndarray = rng.uniform(LO, HI, DIM)
        self.vel: np.ndarray = rng.uniform(-V_MAX, V_MAX, DIM)
        self.fitness: float = f(*self.pos)
        self.best_pos: np.ndarray = self.pos.copy()
        self.best_fit: float = self.fitness


def run_pso(
    n_particles: int,
    max_iter: int,
    c1: float,
    c2: float,
    w: float,
    topology: str = "global",
    speed_strategy: str = "clip",
    seed: int = SEED,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, List[float]]:
    """Run PSO and return (best_position, best_fitness, convergence_curve).

    Args:
        n_particles: swarm size.
        max_iter: maximum iterations.
        c1: cognitive coefficient.
        c2: social coefficient.
        w: inertia weight.
        topology: "global" or "ring".
        speed_strategy: "clip" or "reflect".
        seed: random seed.
        verbose: print progress every 1000 iterations.

    Returns:
        (best_pos, best_fit, [best_fit_per_iteration])
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    speed_fn = clip_speed if speed_strategy == "clip" else reflect_speed

    swarm = [Particle(rng) for _ in range(n_particles)]

    # Global best
    g_best_pos = max(swarm, key=lambda p: p.best_fit).best_pos.copy()
    g_best_fit = max(p.best_fit for p in swarm)

    curve: List[float] = [g_best_fit]

    for it in range(1, max_iter + 1):
        for idx, p in enumerate(swarm):
            # Determine social attractor
            if topology == "global":
                social_pos = g_best_pos
            else:  # ring
                left  = swarm[(idx - 1) % n_particles]
                right = swarm[(idx + 1) % n_particles]
                neighbours = [left, p, right]
                social_pos = max(neighbours, key=lambda q: q.best_fit).best_pos

            r1 = rng.uniform(0, 1, DIM)
            r2 = rng.uniform(0, 1, DIM)

            # Velocity update
            p.vel = (w * p.vel
                     + c1 * r1 * (p.best_pos - p.pos)
                     + c2 * r2 * (social_pos - p.pos))
            p.vel = speed_fn(p.vel)

            # Position update + toroidal wrap
            p.pos = p.pos + p.vel
            for d in range(DIM):
                p.pos[d] = wrap(p.pos[d])

            # Evaluate
            p.fitness = f(*p.pos)
            if p.fitness > p.best_fit:
                p.best_fit = p.fitness
                p.best_pos = p.pos.copy()

            # Update global best
            if p.best_fit > g_best_fit:
                g_best_fit = p.best_fit
                g_best_pos = p.best_pos.copy()

        curve.append(g_best_fit)

        if verbose and it % 2000 == 0:
            print(f"    iter {it:>6}: best = {g_best_fit:.8f}")

    return g_best_pos, g_best_fit, curve


# ----------------------- Part 1: 50 particles parameter tuning ------------

def part1(verbose: bool = True) -> Tuple[float, float, float, float]:
    """Grid-search (c1, c2, w) with 50 particles.

    Returns:
        (best_c1, best_c2, best_w, best_fitness)
    """
    print("=" * 65)
    print("  Exercise 2 — Part 1: PSO Parameter Tuning (50 particles)")
    print("=" * 65)

    c1_vals = [0.5, 1.0, 1.5, 2.0]
    c2_vals = [0.5, 1.0, 1.5, 2.0]
    w_vals  = [0.2, 0.4, 0.6, 0.8, 1.0]

    best_combo = (0.0, 0.0, 0.0)
    best_fitness = -1e30
    all_results: list[Tuple[float, float, float, float]] = []

    for c1, c2, w_val in itertools.product(c1_vals, c2_vals, w_vals):
        _, fit, _ = run_pso(50, MAX_ITER, c1, c2, w_val, seed=SEED)
        all_results.append((c1, c2, w_val, fit))
        if fit > best_fitness:
            best_fitness = fit
            best_combo = (c1, c2, w_val)

    if verbose:
        # Show top-5
        all_results.sort(key=lambda t: t[3], reverse=True)
        print("\n  Top-5 configurations:")
        for c1, c2, w_val, fit in all_results[:5]:
            print(f"    c1={c1:.1f}  c2={c2:.1f}  w={w_val:.1f}  →  f={fit:.8f}")
        print(f"\n  Best: c1*={best_combo[0]:.1f}, c2*={best_combo[1]:.1f}, "
              f"w*={best_combo[2]:.1f}  →  f={best_fitness:.8f}")

    return (*best_combo, best_fitness)


# ----------------------- Part 2: 100 particles with best params -----------

def part2(c1: float, c2: float, w: float, verbose: bool = True) -> List[float]:
    """Run PSO with 100 particles using the winning parameters from Part 1.

    Returns:
        Convergence curve.
    """
    print("\n" + "=" * 65)
    print("  Exercise 2 — Part 2: PSO with 100 particles")
    print("=" * 65)

    pos, fit, curve = run_pso(100, MAX_ITER, c1, c2, w, verbose=verbose)
    if verbose:
        print(f"\n  Best position: x={pos[0]:.6f}, y={pos[1]:.6f}, z={pos[2]:.6f}")
        print(f"  Best fitness:  {fit:.8f}")
    return curve


# ----------------------- Part 3: further search with 100 particles --------

def part3(verbose: bool = True) -> Tuple[Tuple[float, float, float], float, List[float]]:
    """Explore additional (c1, c2, w) with 100 particles.

    Returns:
        (best_triple, best_fitness, convergence_curve)
    """
    print("\n" + "=" * 65)
    print("  Exercise 2 — Part 3: Further Optimisation (100 particles)")
    print("=" * 65)

    c1_vals = [1.0, 1.5, 2.0]
    c2_vals = [1.0, 1.5, 2.0]
    w_vals  = [0.3, 0.5, 0.7, 0.9]

    best_triple = (0.0, 0.0, 0.0)
    best_fitness = -1e30
    best_curve: List[float] = []

    for c1, c2, w_val in itertools.product(c1_vals, c2_vals, w_vals):
        pos, fit, curve = run_pso(100, MAX_ITER, c1, c2, w_val, seed=SEED)
        if fit > best_fitness:
            best_fitness = fit
            best_triple = (c1, c2, w_val)
            best_curve = curve

    if verbose:
        print(f"  Best: c1={best_triple[0]:.1f}, c2={best_triple[1]:.1f}, "
              f"w={best_triple[2]:.1f}  →  f={best_fitness:.8f}")

    return best_triple, best_fitness, best_curve


# ----------------------- topology + speed comparison ----------------------

def topology_speed_comparison(c1: float, c2: float, w: float, verbose: bool = True) -> dict:
    """Compare global vs ring topology and clip vs reflect speed limiting.

    Returns:
        dict of label → convergence curve.
    """
    print("\n" + "=" * 65)
    print("  Exercise 2 — Topology & Speed-limit Comparison")
    print("=" * 65)

    configs = [
        ("Global + Clip",    "global", "clip"),
        ("Global + Reflect", "global", "reflect"),
        ("Ring + Clip",      "ring",   "clip"),
        ("Ring + Reflect",   "ring",   "reflect"),
    ]

    curves: dict[str, List[float]] = {}
    for label, topo, spd in configs:
        pos, fit, curve = run_pso(100, MAX_ITER, c1, c2, w,
                                   topology=topo, speed_strategy=spd, seed=SEED)
        curves[label] = curve
        if verbose:
            print(f"  {label:<22} →  f={fit:.8f}  "
                  f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    return curves


# ----------------------- 2-D slice plot -----------------------------------

def plot_slice_with_swarm(c1: float, c2: float, w: float) -> None:
    """Plot f(x, y, z=z*) colour map with final particle positions.

    Uses 100 particles, global topology.
    """
    rng = np.random.RandomState(SEED)
    random.seed(SEED)

    # Run PSO and collect final positions
    pos_best, fit_best, _ = run_pso(100, MAX_ITER, c1, c2, w, seed=SEED)
    z_star = pos_best[2]

    # Re-run to capture final swarm state (quick, just need final positions)
    # We'll do a short helper:
    rng2 = np.random.RandomState(SEED)
    swarm = [Particle(rng2) for _ in range(100)]
    g_best_pos = max(swarm, key=lambda p: p.best_fit).best_pos.copy()
    g_best_fit = max(p.best_fit for p in swarm)
    for it in range(1, MAX_ITER + 1):
        for p in swarm:
            r1 = rng2.uniform(0, 1, DIM)
            r2 = rng2.uniform(0, 1, DIM)
            p.vel = w * p.vel + c1 * r1 * (p.best_pos - p.pos) + c2 * r2 * (g_best_pos - p.pos)
            p.vel = clip_speed(p.vel)
            p.pos = p.pos + p.vel
            for d in range(DIM):
                p.pos[d] = wrap(p.pos[d])
            p.fitness = f(*p.pos)
            if p.fitness > p.best_fit:
                p.best_fit = p.fitness
                p.best_pos = p.pos.copy()
            if p.best_fit > g_best_fit:
                g_best_fit = p.best_fit
                g_best_pos = p.best_pos.copy()

    final_positions = np.array([p.pos[:2] for p in swarm])

    # Build surface grid
    xs = np.linspace(LO, HI, 300)
    ys = np.linspace(LO, HI, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = f_vec(X, Y, z_star)

    plot_swarm_on_surface(
        X, Y, Z, final_positions,
        f"f(x, y, z*={z_star:.4f}) — Final Swarm",
        os.path.join(RESULTS_DIR, "ex2_swarm_slice.png"),
    )


# ===================== main ===============================================

def main() -> None:
    """Run all PSO experiments."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Part 1
    c1_s, c2_s, w_s, _ = part1(verbose=True)

    # Part 2
    curve_p2 = part2(c1_s, c2_s, w_s, verbose=True)

    # Part 3
    (c1_3, c2_3, w_3), _, curve_p3 = part3(verbose=True)

    # Plot Part 2 vs Part 3 convergence
    plot_convergence(
        {"Part 2 (50→100, best)": curve_p2, "Part 3 (100, further)": curve_p3},
        "Exercise 2 — PSO Convergence (Part 2 vs 3)",
        os.path.join(RESULTS_DIR, "ex2_convergence_p2_p3.png"),
        xlabel="Iteration",
        ylabel="Best Fitness",
    )

    # Topology + speed comparison
    comp_curves = topology_speed_comparison(c1_s, c2_s, w_s, verbose=True)
    plot_convergence(
        comp_curves,
        "Exercise 2 — Topology & Speed-limit Comparison",
        os.path.join(RESULTS_DIR, "ex2_topology_speed.png"),
        xlabel="Iteration",
        ylabel="Best Fitness",
    )

    # 2-D slice with swarm
    print("\n  Generating 2-D slice plot …")
    plot_slice_with_swarm(c1_s, c2_s, w_s)

    print("\n  All done — plots saved to results/")


if __name__ == "__main__":
    main()
