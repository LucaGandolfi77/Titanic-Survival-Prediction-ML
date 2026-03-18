"""Exercise 2A — Pattern Guessing GA.

Evolves a binary individual to match a target pattern loaded from smiley.txt.
Fitness = number of matching bits (maximize).
"""

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import plot_grid_pattern, plot_convergence_single


def load_target(filepath):
    """Load target pattern from a text file.

    Args:
        filepath: path to the text file with space-separated 0/1 values.

    Returns:
        Tuple of (flat numpy array of ints, number of rows, number of columns).
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split()])
    grid = np.array(rows, dtype=int)
    return grid.flatten(), grid.shape[0], grid.shape[1]


def make_eval_func(target):
    """Create an evaluation function that counts matching bits.

    Args:
        target: flat numpy array of the target pattern.

    Returns:
        Evaluation function compatible with DEAP.
    """
    nf_counter = [0]  # mutable counter for exact Nf tracking

    def eval_pattern(individual):
        """Count the number of bits matching the target.

        Args:
            individual: list/array of 0/1 integers.

        Returns:
            Tuple with the number of matching bits.
        """
        nf_counter[0] += 1
        ind_arr = np.array(individual, dtype=int)
        matches = int(np.sum(ind_arr == target))
        return (matches,)

    return eval_pattern, nf_counter


def run(verbose=True):
    """Run the pattern guessing GA for Exercise 2A.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (best_individual, best_fitness, logbook, nf_total, rows, cols).
    """
    random.seed(42)
    np.random.seed(42)

    # Load target
    smiley_path = os.path.join(os.path.dirname(__file__), "smiley.txt")
    target, rows, cols = load_target(smiley_path)
    n_bits = len(target)

    if verbose:
        print("=" * 60)
        print(f"Exercise 2A — Pattern Guessing ({cols}x{rows}, {n_bits} bits)")
        print("=" * 60)

    # --- DEAP setup ---
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bit, n=n_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_func, nf_counter = make_eval_func(target)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_bits)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Run GA ---
    pop_size = 300
    ngen = 200
    cxpb = 0.7
    mutpb = 0.2

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)

    best = hof[0]
    best_fit = best.fitness.values[0]
    nf_total = nf_counter[0]

    if verbose:
        print(f"\n  Best fitness (matching bits): {best_fit}/{n_bits}")
        print(f"  Total Nf (exact): {nf_total}")
        print(f"  Perfect match: {'YES' if best_fit == n_bits else 'NO'}")

    # Visualize
    output_dir = os.path.dirname(__file__)
    plot_grid_pattern(best, rows, cols, f"Best Pattern ({best_fit}/{n_bits} bits)",
                      os.path.join(output_dir, "exercise2_best_pattern.png"))
    plot_grid_pattern(target, rows, cols, "Target Pattern",
                      os.path.join(output_dir, "exercise2_target_pattern.png"))
    plot_convergence_single(logbook, "Exercise 2A — Pattern Guessing Convergence",
                            os.path.join(output_dir, "exercise2_convergence.png"),
                            best_key="max")

    return best, best_fit, logbook, nf_total, rows, cols


if __name__ == "__main__":
    run(verbose=True)
