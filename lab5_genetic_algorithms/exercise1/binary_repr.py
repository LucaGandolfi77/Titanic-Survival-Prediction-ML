"""Exercise 1B — Function Optimization with Binary Representation.

Minimizes f(x, y, z) = (1.5 + sin(z)) * (sqrt((20 - x)^2 + (30 - y)^2) + 1)
using a GA with individuals represented as 90 bits (30 bits per variable).
Each 30-bit chunk is decoded to a float in [-250, 250].
"""

import random
import math
import numpy as np
from deap import base, creator, tools, algorithms

# Encoding parameters
BITS_PER_VAR = 30
NUM_VARS = 3
TOTAL_BITS = BITS_PER_VAR * NUM_VARS
VAR_MIN = -250.0
VAR_MAX = 250.0
MAX_INT = 2 ** BITS_PER_VAR - 1


def decode_individual(individual):
    """Decode a binary individual into a list of float variables.

    Each 30-bit chunk is converted to an unsigned integer, then mapped to [VAR_MIN, VAR_MAX].

    Args:
        individual: list of 90 bits (0/1 integers).

    Returns:
        List of 3 float values [x, y, z].
    """
    values = []
    for v in range(NUM_VARS):
        start = v * BITS_PER_VAR
        bits = individual[start:start + BITS_PER_VAR]
        # Convert bit list to unsigned integer
        int_val = 0
        for bit in bits:
            int_val = (int_val << 1) | bit
        # Map to float range
        float_val = VAR_MIN + int_val / MAX_INT * (VAR_MAX - VAR_MIN)
        values.append(float_val)
    return values


def eval_func(individual):
    """Evaluate the objective function using the decoded float values.

    Args:
        individual: list of 90 bits.

    Returns:
        Tuple with the fitness value.
    """
    x, y, z = decode_individual(individual)
    distance = math.sqrt((20 - x) ** 2 + (30 - y) ** 2) + 1
    fitness = (1.5 + math.sin(z)) * distance
    return (fitness,)


def run(verbose=True):
    """Run the binary-representation GA for Exercise 1B.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (best_individual_decoded, best_fitness, logbook, hall_of_fame).
    """
    random.seed(42)
    np.random.seed(42)

    # --- DEAP setup ---
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bit, n=TOTAL_BITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / TOTAL_BITS)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Run GA ---
    pop_size = 200
    ngen = 300
    cxpb = 0.7
    mutpb = 0.2

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    if verbose:
        print("=" * 60)
        print("Exercise 1B — Binary Representation")
        print("=" * 60)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)

    best = hof[0]
    best_decoded = decode_individual(best)
    best_fit = best.fitness.values[0]
    nf = pop_size + ngen * pop_size

    if verbose:
        print(f"\n  Best individual (decoded): x={best_decoded[0]:.6f}, "
              f"y={best_decoded[1]:.6f}, z={best_decoded[2]:.6f}")
        print(f"  Best fitness: {best_fit:.6f}")
        print(f"  Total Nf (approx): {nf}")

    return best_decoded, best_fit, logbook, hof


if __name__ == "__main__":
    run(verbose=True)
