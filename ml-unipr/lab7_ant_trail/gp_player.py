"""gp_player.py — Genetic Programming player for the Ant Trail Problem.

Uses DEAP to evolve a symbolic expression that maps the flattened
(2m+1)² neighbourhood to a real value.  A threshold scheme converts
this value to a direction:

    output ≤ −K  → up
    −K < output ≤ 0  → down
     0 < output ≤ K  → right
    output > K  → left

Fitness is the total score accumulated over a set of training games.
"""

from __future__ import annotations

import operator
import random
from typing import Any

import numpy as np
from deap import base, creator, gp, tools, algorithms

from board import Board
from ant import Ant, DIRECTIONS, OPPOSITE

# -----------------------------------------------------------------------
# Protected operators
# -----------------------------------------------------------------------

def _random_erc() -> float:
    """ERC generator for DEAP (named function to allow pickling)."""
    return round(random.uniform(-2, 2), 2)

def _protected_div(a: float, b: float) -> float:
    """Protected division: returns 1.0 when dividing by (near-)zero."""
    try:
        return a / b if abs(b) > 1e-6 else 1.0
    except (ZeroDivisionError, OverflowError):
        return 1.0


def _neg(a: float) -> float:
    return -a


# -----------------------------------------------------------------------
# GP ↔ direction mapping
# -----------------------------------------------------------------------

_K = 1.0

_DIR_ORDER = ["up", "down", "right", "left"]


def _output_to_dir(val: float) -> str:
    if val <= -_K:
        return "up"
    elif val <= 0:
        return "down"
    elif val <= _K:
        return "right"
    else:
        return "left"


_DIR_TO_LABEL = {"up": 0, "down": 1, "right": 2, "left": 3}


# -----------------------------------------------------------------------
# Build DEAP primitive set
# -----------------------------------------------------------------------

def _make_pset(n_features: int) -> gp.PrimitiveSet:
    """Build a typed primitive set for *n_features* float terminals."""
    pset = gp.PrimitiveSet("ANT_GP", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(_neg, 1)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(max, 2)
    pset.addEphemeralConstant("ERC", _random_erc)
    return pset


# -----------------------------------------------------------------------
# Fitness evaluation
# -----------------------------------------------------------------------

def _evaluate(individual, pset, n, m, seeds, n_games):
    """Evaluate a GP individual by playing *n_games* on seeded boards."""
    func = gp.compile(individual, pset)
    total_score = 0

    for seed in seeds[:n_games]:
        board = Board(n, seed=seed)
        rng = np.random.RandomState(seed + 1000)
        empty = [(r, c) for r in range(n) for c in range(n)
                 if board.grid[r, c] == 0]
        if not empty:
            empty = [(r, c) for r in range(n) for c in range(n)]
        sr, sc = empty[rng.randint(len(empty))]
        ant = Ant(board, sr, sc, m=m)

        random.seed(seed + 2000)

        while ant.has_moves_left():
            nb = ant.get_neighborhood()
            args = tuple(float(x) for x in nb)
            try:
                out = float(func(*args))
            except Exception:
                out = 0.0

            # Fallback rule
            if np.all(nb <= 0):
                choices = [d for d in DIRECTIONS
                           if d != OPPOSITE.get(ant.last_dir, "")]
                if not choices:
                    choices = list(DIRECTIONS)
                direction = random.choice(choices)
            else:
                direction = _output_to_dir(out)

            ant.move(direction)

        total_score += ant.score

    return (total_score,)


# -----------------------------------------------------------------------
# Training entry point
# -----------------------------------------------------------------------

# Create DEAP fitness / individual classes only once
if not hasattr(creator, "FitnessMax_GP"):
    creator.create("FitnessMax_GP", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual_GP"):
    creator.create("Individual_GP", gp.PrimitiveTree,
                   fitness=creator.FitnessMax_GP)


def train_gp(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    n: int = 10,
    m: int = 1,
    pop_size: int = 200,
    n_gen: int = 50,
    n_games: int = 10,
    base_seed: int = 42,
    verbose: bool = True,
) -> Any:
    """Evolve a GP individual for the ant game.

    Training is done via *game simulation*, not supervised learning on
    (X, y) data (though those arguments are accepted for API
    compatibility with ant_train).

    Args:
        X, y: ignored (present for interface compatibility).
        n: board size.
        m: neighbourhood width.
        pop_size: population size.
        n_gen: number of generations.
        n_games: how many games per fitness evaluation.
        base_seed: base seed for training boards.
        verbose: print evolution stats.

    Returns:
        A compiled function (callable) mapping *n_features* floats →
        float, with a ``._individual`` attribute storing the raw tree.
    """
    n_features = (2 * m + 1) ** 2
    pset = _make_pset(n_features)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual_GP, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    seeds = [base_seed + i for i in range(n_games)]
    toolbox.register("evaluate", _evaluate, pset=pset, n=n, m=m,
                     seeds=seeds, n_games=n_games)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                            max_value=12))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=12))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.7, mutpb=0.2, ngen=n_gen,
                        stats=stats if verbose else None,
                        halloffame=hof, verbose=verbose)

    best = hof[0]
    func = gp.compile(best, pset)

    # Attach metadata
    func._individual = best
    func._pset = pset

    if verbose:
        print(f"  [GP] Best fitness: {best.fitness.values[0]}")
        print(f"  [GP] Tree height : {best.height}, nodes: {len(best)}")

    return func


# -----------------------------------------------------------------------
# Standalone
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GP ant player")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("-m", type=int, default=1)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    func = train_gp(n=args.n, m=args.m, pop_size=args.pop,
                     n_gen=args.gen, n_games=args.games,
                     base_seed=args.seed, verbose=True)
    print(f"GP player ready: {func._individual}")
