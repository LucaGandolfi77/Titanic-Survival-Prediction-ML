"""Exercise 1A — Function Optimization with Float Representation.

Minimizes f(x, y, z) = (1.5 + sin(z)) * (sqrt((20 - x)^2 + (30 - y)^2) + 1)
using a GA with individuals represented as lists of 3 floats in [-250, 250].
"""

import random
import math
import numpy as np
from deap import base, creator, tools, algorithms


def eval_func(individual):
    """Evaluate the objective function f(x, y, z).

    Args:
        individual: list of 3 floats [x, y, z].

    Returns:
        Tuple with the fitness value (required by DEAP).
    """
    x, y, z = individual
    distance = math.sqrt((20 - x) ** 2 + (30 - y) ** 2) + 1
    fitness = (1.5 + math.sin(z)) * distance
    return (fitness,)


def clamp(individual, lo=-250, hi=250):
    """Clamp all genes of an individual to [lo, hi].

    Args:
        individual: list of floats.
        lo: lower bound.
        hi: upper bound.

    Returns:
        The clamped individual.
    """
    for i in range(len(individual)):
        individual[i] = max(lo, min(hi, individual[i]))
    return individual


def run(verbose=True):
    """Run the float-representation GA for Exercise 1A.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (best_individual, best_fitness, logbook, hall_of_fame).
    """
    random.seed(42)
    np.random.seed(42)

    # --- DEAP setup ---
    # Clean up any previous creator definitions
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -250, 250)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Decorator to clamp values after crossover/mutation
    def clamp_decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                clamp(ind)
                del ind.fitness.values
            return offspring
        return wrapper

    toolbox.decorate("mate", clamp_decorator)
    toolbox.decorate("mutate", clamp_decorator)

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
        print("Exercise 1A — Float Representation")
        print("=" * 60)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                       ngen=ngen, stats=stats, halloffame=hof,
                                       verbose=verbose)

    best = hof[0]
    best_fit = best.fitness.values[0]

    # Compute total fitness evaluations: initial pop + ngen * (cxpb * pop_size + mutpb * pop_size) approx
    # eaSimple evaluates all offspring each generation; offspring size = pop_size
    nf = pop_size + ngen * pop_size  # upper bound for eaSimple

    if verbose:
        print(f"\n  Best individual: x={best[0]:.6f}, y={best[1]:.6f}, z={best[2]:.6f}")
        print(f"  Best fitness:    {best_fit:.6f}")
        print(f"  Total Nf (approx): {nf}")

    return best, best_fit, logbook, hof


if __name__ == "__main__":
    run(verbose=True)
