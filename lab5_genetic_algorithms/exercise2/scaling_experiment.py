"""Exercise 2B — Scaling Experiment and Hyperparameter Search.

Part 1: Measures how average Nf scales with pattern size.
Part 2: Finds the best (pop_size, ngen) combination for the smiley target.
"""

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import plot_scaling, plot_heatmap


def generate_random_pattern(rows, cols, seed=None):
    """Generate a random binary pattern.

    Args:
        rows: number of rows.
        cols: number of columns.
        seed: random seed for reproducibility.

    Returns:
        Flat numpy array of 0/1 integers.
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=rows * cols).astype(int)


def run_pattern_ga(target, pop_size, ngen, cxpb=0.7, mutpb=0.2, seed=42):
    """Run the pattern-matching GA and return exact Nf and whether perfect fitness was reached.

    Args:
        target: flat numpy array of 0/1 target pattern.
        pop_size: population size.
        ngen: number of generations.
        cxpb: crossover probability.
        mutpb: mutation probability.
        seed: random seed.

    Returns:
        Tuple of (nf_total, reached_perfect, best_fitness).
    """
    random.seed(seed)
    np.random.seed(seed)

    n_bits = len(target)
    nf_counter = [0]

    def eval_pattern(individual):
        """Count matching bits."""
        nf_counter[0] += 1
        ind_arr = np.array(individual, dtype=int)
        return (int(np.sum(ind_arr == target)),)

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
    toolbox.register("evaluate", eval_pattern)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_bits)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=False)

    best_fit = hof[0].fitness.values[0]
    reached_perfect = (best_fit == n_bits)
    return nf_counter[0], reached_perfect, best_fit


def run_scaling_experiment(verbose=True):
    """Part 1: Measure average Nf vs pattern size.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (sizes_list, avg_nf_list).
    """
    if verbose:
        print("=" * 60)
        print("Exercise 2B — Scaling Experiment (Nf vs pattern size)")
        print("=" * 60)

    # Pattern sizes: (rows, cols)
    pattern_configs = [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (16, 14)]
    n_runs = 5
    pop_size = 300
    ngen = 500  # generous limit

    sizes = []
    avg_nfs = []

    for rows, cols in pattern_configs:
        n_bits = rows * cols
        sizes.append(n_bits)
        nf_values = []

        for run_i in range(n_runs):
            target = generate_random_pattern(rows, cols, seed=run_i * 100)
            nf, perfect, best_fit = run_pattern_ga(
                target, pop_size=pop_size, ngen=ngen, seed=42 + run_i
            )
            nf_values.append(nf)
            if verbose:
                status = "PERFECT" if perfect else f"{best_fit}/{n_bits}"
                print(f"  Size {rows}x{cols} ({n_bits} bits), run {run_i+1}: "
                      f"Nf={nf}, {status}")

        avg_nf = np.mean(nf_values)
        avg_nfs.append(avg_nf)
        if verbose:
            print(f"  -> Average Nf for {rows}x{cols}: {avg_nf:.0f}\n")

    # Plot
    output_dir = os.path.dirname(__file__)
    plot_scaling(sizes, avg_nfs, "Nf vs Pattern Size",
                 os.path.join(output_dir, "exercise2_scaling.png"))

    return sizes, avg_nfs


def run_hyperparameter_search(verbose=True):
    """Part 2: Find the best (pop_size, ngen) combination for smiley.txt.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (best_pop, best_ngen, nf_matrix).
    """
    if verbose:
        print("=" * 60)
        print("Exercise 2B — Hyperparameter Search (pop_size × ngen)")
        print("=" * 60)

    # Load smiley target
    smiley_path = os.path.join(os.path.dirname(__file__), "smiley.txt")
    rows_data = []
    with open(smiley_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows_data.append([int(x) for x in line.split()])
    target = np.array(rows_data, dtype=int).flatten()

    pop_sizes = [50, 100, 200, 300, 500]
    gen_counts = [50, 100, 200, 300]
    n_runs = 5

    nf_matrix = np.full((len(gen_counts), len(pop_sizes)), np.inf)

    for i, ngen in enumerate(gen_counts):
        for j, pop_sz in enumerate(pop_sizes):
            nf_values = []
            for run_i in range(n_runs):
                nf, perfect, best_fit = run_pattern_ga(
                    target, pop_size=pop_sz, ngen=ngen, seed=42 + run_i
                )
                nf_values.append(nf)

            avg_nf = np.mean(nf_values)
            nf_matrix[i, j] = avg_nf
            if verbose:
                print(f"  pop={pop_sz:>4}, ngen={ngen:>4} -> avg Nf={avg_nf:.0f}")

    # Find best combination (lowest avg Nf that still reaches high fitness)
    best_idx = np.unravel_index(np.argmin(nf_matrix), nf_matrix.shape)
    best_ngen = gen_counts[best_idx[0]]
    best_pop = pop_sizes[best_idx[1]]

    if verbose:
        print(f"\n  Best combination: pop_size={best_pop}, ngen={best_ngen}")
        print(f"  Avg Nf: {nf_matrix[best_idx[0], best_idx[1]]:.0f}")

    # Plot heatmap
    output_dir = os.path.dirname(__file__)
    plot_heatmap(pop_sizes, gen_counts, nf_matrix,
                 "Avg Nf — Hyperparameter Search",
                 os.path.join(output_dir, "exercise2_heatmap.png"))

    return best_pop, best_ngen, nf_matrix


def run(verbose=True):
    """Run both parts of the scaling experiment.

    Args:
        verbose: if True, print progress info.

    Returns:
        Dict with results from both parts.
    """
    sizes, avg_nfs = run_scaling_experiment(verbose=verbose)
    best_pop, best_ngen, nf_matrix = run_hyperparameter_search(verbose=verbose)
    return {
        "scaling": (sizes, avg_nfs),
        "hyperparams": (best_pop, best_ngen, nf_matrix),
    }


if __name__ == "__main__":
    run(verbose=True)
