#!/usr/bin/env python3
"""
Exercise 0a — Symbolic Regression via Genetic Programming
=========================================================

**Design choices:**

Target function:
  f(x, y) = sin(x) * cos(y) + log(1 + x^2) * exp(-y^2 / 10)

This mixes transcendental (sin, cos, exp) and logarithmic components, so it
CANNOT be exactly reproduced by the finite function set {+, -, *, pdiv, sin, cos}.
The exp(·) and log(·) terms force the GP to *approximate* rather than replicate.

Function set F = {add, sub, mul, protected_div, sin, cos}
Terminal set T = {x, y, ERC_i}   with  ERC_i ~ U(-10, 10)

Sampling domain: x ∈ [-5, 5], y ∈ [-5, 5], 50×50 grid (2 500 fitness cases).
Fitness:  MSE(predicted, true)  →  MINIMIZE.

After the main run we also sweep population size, max depth, crossover/mutation
probabilities, and generation count, saving convergence curves for each.
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import List, Tuple

import numpy as np # pyright: ignore[reportMissingImports]
from deap import algorithms, base, creator, gp, tools # pyright: ignore[reportMissingImports]

# --- local imports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    RESULTS_DIR,
    protected_div,
    plot_surface_comparison,
    plot_convergence,
)

# ===================== HYPERPARAMETERS ====================================
POP_SIZE          = 500
NGEN              = 80
CXPB              = 0.7
MUTPB             = 0.2
MAX_DEPTH_INIT    = 4       # ramped half-and-half max depth
MAX_DEPTH_LIMIT   = 12      # bloat control
TOURN_SIZE        = 5
SEED              = 42
DOMAIN_LO, DOMAIN_HI = -5.0, 5.0
GRID_N            = 50      # 50×50 = 2500 points
# ==========================================================================

# ----------------------- target function ----------------------------------

def target_function(x: float, y: float) -> float:
    """Ground-truth function that GP must approximate.

    f(x, y) = sin(x)*cos(y) + log(1 + x^2)*exp(-y^2 / 10)
    """
    return math.sin(x) * math.cos(y) + math.log(1.0 + x * x) * math.exp(-y * y / 10.0)


# ----------------------- data set -----------------------------------------

def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 50×50 evaluation grid and the corresponding target values.

    Returns:
        (X, Y, Z_true) — meshgrid arrays of shape (GRID_N, GRID_N).
    """
    xs = np.linspace(DOMAIN_LO, DOMAIN_HI, GRID_N)
    ys = np.linspace(DOMAIN_LO, DOMAIN_HI, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    vfunc = np.vectorize(target_function)
    Z = vfunc(X, Y)
    return X, Y, Z


# ----------------------- GP primitives ------------------------------------

def _setup_pset() -> gp.PrimitiveSetTyped | gp.PrimitiveSet:
    """Create the DEAP primitive set."""
    pset = gp.PrimitiveSet("MAIN", arity=2)  # x, y
    pset.addPrimitive(np.add,        2, name="add")
    pset.addPrimitive(np.subtract,   2, name="sub")
    pset.addPrimitive(np.multiply,   2, name="mul")
    pset.addPrimitive(protected_div, 2, name="pdiv")
    pset.addPrimitive(math.sin,      1, name="sin")
    pset.addPrimitive(math.cos,      1, name="cos")
    pset.addEphemeralConstant("ERC", lambda: random.uniform(-10.0, 10.0))
    pset.renameArguments(ARG0="x", ARG1="y")
    return pset


# ----------------------- evaluation ---------------------------------------

def _eval_symbreg(individual, toolbox, X, Y, Z_true):
    """Evaluate a GP tree on the entire grid, returning (MSE,).

    Args:
        individual: DEAP GP tree.
        toolbox: must contain ``compile`` method.
        X, Y, Z_true: meshgrid arrays.

    Returns:
        Tuple[float]: (mse_value,)
    """
    func = toolbox.compile(expr=individual)
    total_err = 0.0
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                pred = func(X[i, j], Y[i, j])
                if not np.isfinite(pred):
                    pred = 0.0
            except (OverflowError, ValueError, ZeroDivisionError):
                pred = 0.0
            total_err += (pred - Z_true[i, j]) ** 2
            count += 1
    return (total_err / count,)


# ----------------------- core GA runner -----------------------------------

def run_gp(
    pop_size: int = POP_SIZE,
    ngen: int = NGEN,
    cxpb: float = CXPB,
    mutpb: float = MUTPB,
    max_depth: int = MAX_DEPTH_LIMIT,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[object, tools.Logbook, tools.HallOfFame]:
    """Run a single GP experiment and return best individual + logbook.

    Args:
        pop_size: population size.
        ngen: number of generations.
        cxpb: crossover probability.
        mutpb: mutation probability.
        max_depth: bloat-control limit.
        seed: random seed.
        verbose: print per-generation stats.

    Returns:
        (best_individual, logbook, hall_of_fame)
    """
    random.seed(seed)
    np.random.seed(seed)

    X, Y, Z_true = make_dataset()
    pset = _setup_pset()

    # --- creator (clean up previous definitions) ---
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
    toolbox.register("evaluate", _eval_symbreg, toolbox=toolbox, X=X, Y=Y, Z_true=Z_true)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=max_depth))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)
    return hof[0], logbook, hof


# ----------------------- parameter sweep ----------------------------------

def parameter_sweep(verbose: bool = False) -> dict:
    """Run a small grid of experiments to study hyper-parameter effects.

    Returns:
        dict mapping config-label → list-of-best-min-per-generation.
    """
    curves: dict[str, list[float]] = {}

    configs = [
        ("pop=200",   dict(pop_size=200,  ngen=60, cxpb=0.7, mutpb=0.2, max_depth=12)),
        ("pop=500",   dict(pop_size=500,  ngen=60, cxpb=0.7, mutpb=0.2, max_depth=12)),
        ("depth=8",   dict(pop_size=500,  ngen=60, cxpb=0.7, mutpb=0.2, max_depth=8)),
        ("depth=17",  dict(pop_size=500,  ngen=60, cxpb=0.7, mutpb=0.2, max_depth=17)),
        ("cx=0.9",    dict(pop_size=500,  ngen=60, cxpb=0.9, mutpb=0.1, max_depth=12)),
        ("mut=0.4",   dict(pop_size=500,  ngen=60, cxpb=0.5, mutpb=0.4, max_depth=12)),
        ("ngen=120",  dict(pop_size=500,  ngen=120, cxpb=0.7, mutpb=0.2, max_depth=12)),
    ]

    for label, kw in configs:
        print(f"\n--- sweep: {label} ---")
        _, logbook, _ = run_gp(**kw, seed=SEED, verbose=verbose)
        curves[label] = logbook.select("min")

    return curves


# ===================== main ===============================================

def main() -> None:
    """Entry point: run GP, plot results, perform hyper-parameter sweep."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  Exercise 0a — Symbolic Regression via GP")
    print("=" * 65)

    # 1. Main run
    best, logbook, hof = run_gp(verbose=True)
    best_mse = best.fitness.values[0]
    print(f"\n  Best tree:    {best}")
    print(f"  Best MSE:     {best_mse:.6f}")
    print(f"  Tree height:  {best.height}")

    # 2. Plot true vs predicted
    X, Y, Z_true = make_dataset()
    pset = _setup_pset()
    func = gp.compile(best, pset)
    Z_pred = np.zeros_like(Z_true)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                val = func(X[i, j], Y[i, j])
                if not np.isfinite(val):
                    val = 0.0
            except Exception:
                val = 0.0
            Z_pred[i, j] = val

    plot_surface_comparison(
        X, Y, Z_true, Z_pred,
        f"Symbolic Regression — MSE = {best_mse:.4f}",
        os.path.join(RESULTS_DIR, "ex0a_surfaces.png"),
    )

    # 3. Convergence plot for main run
    plot_convergence(
        {"Best MSE": logbook.select("min"), "Avg MSE": logbook.select("avg")},
        "Exercise 0a — GP Convergence",
        os.path.join(RESULTS_DIR, "ex0a_convergence.png"),
        ylabel="MSE",
    )

    # 4. Hyper-parameter sweep
    print("\n>>> Hyper-parameter sweep …")
    curves = parameter_sweep(verbose=False)
    plot_convergence(
        curves,
        "Exercise 0a — Hyper-parameter Sweep (best MSE)",
        os.path.join(RESULTS_DIR, "ex0a_sweep.png"),
        ylabel="Best MSE",
    )

    print("\n  Done — all plots saved to results/")


if __name__ == "__main__":
    main()
