"""Exercise 3B — N-Queens with Dumb (2*N Integer-Pair) Representation.

Representation: list of 2*N integers in [0, N-1] encoding N (row, col) pairs.
No constraints on uniqueness — queens can share rows, columns, and diagonals.
Fitness: minimize total conflicts (row + column + diagonal).
"""

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import plot_chessboard


def count_all_conflicts(individual, n):
    """Count all conflicts (row, column, diagonal) among N queens.

    Args:
        individual: list of 2*N integers. Pairs: (individual[2*i], individual[2*i+1]) = (row, col).
        n: board size.

    Returns:
        Tuple with the total number of conflicts.
    """
    queens = [(individual[2 * i], individual[2 * i + 1]) for i in range(n)]
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            r1, c1 = queens[i]
            r2, c2 = queens[j]
            # Same row
            if r1 == r2:
                conflicts += 1
            # Same column
            if c1 == c2:
                conflicts += 1
            # Same diagonal
            if abs(r1 - r2) == abs(c1 - c2):
                conflicts += 1
    return (conflicts,)


def run_single(n, pop_size=300, ngen=100, cxpb=0.5, mutpb=0.2, seed=42, verbose=False):
    """Run the dumb N-Queens GA for a single board size.

    Args:
        n: board size.
        pop_size: population size.
        ngen: number of generations.
        cxpb: crossover probability.
        mutpb: mutation probability.
        seed: random seed.
        verbose: if True, print GA generation info.

    Returns:
        Tuple of (best_queens_as_permutation_or_None, best_fitness, solved).
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- DEAP setup ---
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, n - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, n=2 * n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", count_all_conflicts, n=n)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=n - 1, indpb=1.0 / (2 * n))
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)

    best = hof[0]
    best_fit = best.fitness.values[0]
    solved = (best_fit == 0)

    # Extract queen positions for visualization
    queens = [(best[2 * i], best[2 * i + 1]) for i in range(n)]

    return queens, best_fit, solved


def run(verbose=True):
    """Run Exercise 3B: test dumb representation for increasing N.

    Args:
        verbose: if True, print progress info.

    Returns:
        Dict with N values, success rates, and best solutions.
    """
    if verbose:
        print("=" * 60)
        print("Exercise 3B — N-Queens (Dumb / 2*N Integer-Pair Representation)")
        print("=" * 60)

    n_values = [8, 16, 32, 64, 128]
    n_runs = 10
    results = {}

    for n in n_values:
        successes = 0
        best_queens = None
        best_fit_overall = float("inf")

        for run_i in range(n_runs):
            queens, fit, solved = run_single(n, seed=42 + run_i, verbose=False)
            if solved:
                successes += 1
            if fit < best_fit_overall:
                best_fit_overall = fit
                best_queens = queens

        rate = successes / n_runs * 100
        # Search space: smart = N!, dumb = N^(2*N)
        import math
        smart_space = math.factorial(n)
        dumb_space = n ** (2 * n)
        # Use logarithms to avoid OverflowError for large N
        log_ratio = 2 * n * math.log10(n) - sum(math.log10(i) for i in range(1, n + 1))
        ratio = 10 ** min(log_ratio, 308)  # clamp to float range

        results[n] = {
            "success_rate": rate,
            "best_queens": best_queens,
            "best_fitness": best_fit_overall,
            "search_space_ratio": ratio,
        }

        if verbose:
            print(f"  N={n:>4}: success rate = {rate:5.1f}% "
                  f"({successes}/{n_runs}), best fitness = {best_fit_overall}, "
                  f"space ratio = {ratio:.2e}")

    return results


def print_comparison(smart_results, dumb_results, verbose=True):
    """Print a comparison table of smart vs dumb representations.

    Args:
        smart_results: dict from nqueens_smart.run().
        dumb_results: dict from nqueens_dumb.run().
        verbose: if True, print the table.
    """
    if not verbose:
        return

    import math
    print("\n" + "=" * 80)
    print("Comparison: Smart vs Dumb Representation")
    print("=" * 80)
    print(f"{'N':>5} | {'Smart %':>10} | {'Dumb %':>10} | {'Space Ratio':>15}")
    print("-" * 50)

    all_ns = sorted(set(list(smart_results.keys()) + list(dumb_results.keys())))
    for n in all_ns:
        s_rate = smart_results.get(n, {}).get("success_rate", 0)
        d_rate = dumb_results.get(n, {}).get("success_rate", 0)
        ratio = dumb_results.get(n, {}).get("search_space_ratio", 0)
        print(f"{n:>5} | {s_rate:>9.1f}% | {d_rate:>9.1f}% | {ratio:>15.2e}")


if __name__ == "__main__":
    results = run(verbose=True)
