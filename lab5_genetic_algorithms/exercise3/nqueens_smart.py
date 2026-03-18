"""Exercise 3A — N-Queens with Smart (Permutation) Representation.

Representation: permutation of [0, 1, ..., N-1] where individual[i] = column of queen in row i.
This guarantees no two queens share a row or column.
Fitness: minimize the number of diagonal conflicts.
"""

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import plot_chessboard


def count_diagonal_conflicts(individual):
    """Count the number of pairs of queens that share a diagonal.

    Args:
        individual: permutation list where individual[i] = column of queen in row i.

    Returns:
        Tuple with the number of diagonal conflicts.
    """
    n = len(individual)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(individual[i] - individual[j]):
                conflicts += 1
    return (conflicts,)


def run_single(n, pop_size=300, ngen=100, cxpb=0.5, mutpb=0.2, seed=42, verbose=False):
    """Run the smart N-Queens GA for a single board size.

    Args:
        n: board size.
        pop_size: population size.
        ngen: number of generations.
        cxpb: crossover probability.
        mutpb: mutation probability.
        seed: random seed.
        verbose: if True, print GA generation info.

    Returns:
        Tuple of (best_individual, best_fitness, solved).
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
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", count_diagonal_conflicts)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / n)
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
    return list(best), best_fit, solved


def run(verbose=True):
    """Run Exercise 3A: test smart representation for increasing N.

    Args:
        verbose: if True, print progress info.

    Returns:
        Dict with N values, success rates, and best solutions.
    """
    if verbose:
        print("=" * 60)
        print("Exercise 3A — N-Queens (Smart / Permutation Representation)")
        print("=" * 60)

    n_values = [8, 16, 32, 64, 128]
    n_runs = 10
    results = {}

    for n in n_values:
        successes = 0
        best_solution = None
        best_fit_overall = float("inf")

        for run_i in range(n_runs):
            sol, fit, solved = run_single(n, seed=42 + run_i, verbose=False)
            if solved:
                successes += 1
            if fit < best_fit_overall:
                best_fit_overall = fit
                best_solution = sol

        rate = successes / n_runs * 100
        results[n] = {
            "success_rate": rate,
            "best_solution": best_solution,
            "best_fitness": best_fit_overall,
        }

        if verbose:
            print(f"  N={n:>4}: success rate = {rate:5.1f}% "
                  f"({successes}/{n_runs}), best fitness = {best_fit_overall}")

    # Find largest N with success rate > 50%
    largest_n = max((n for n in n_values if results[n]["success_rate"] > 50), default=0)
    if verbose:
        print(f"\n  Largest N with success rate > 50%: {largest_n}")

    # Visualize best solution for N=8
    output_dir = os.path.dirname(__file__)
    if 8 in results and results[8]["best_solution"] is not None:
        plot_chessboard(results[8]["best_solution"], 8,
                        f"N-Queens Smart (N=8, conflicts={results[8]['best_fitness']:.0f})",
                        os.path.join(output_dir, "exercise3_smart_8queens.png"))

    return results


if __name__ == "__main__":
    run(verbose=True)
