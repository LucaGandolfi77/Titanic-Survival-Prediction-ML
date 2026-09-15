import random
import numpy as np
import sys

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generate a bit: 0 or 1
toolbox.register("attr_bool", random.randint, 0, 1)

# Individual = 90 bits
toolbox.register("individual", tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_bool,
                 90)

# Population
toolbox.register("population", tools.initRepeat,
                 list,
                 toolbox.individual)


# ---------------------------------------------------------
# Decode 30 bits into a value in [-250, 250]
# ---------------------------------------------------------

def decode(bits):
    value = 0

    for bit in bits:
        value = value * 2 + bit

    # 30 bits -> integer in [0, 2^30 - 1]
    max_value = 2**30 - 1

    # map [0, max_value] -> [-250, 250]
    return -250 + (500 * value / max_value)


def eval_function(individual):

    # First 30 bits -> x
    x = decode(individual[0:30])

    # Next 30 bits -> y
    y = decode(individual[30:60])

    # Last 30 bits -> z
    z = decode(individual[60:90])

    val = (1.5 + np.sin(z)) * (
        np.sqrt((20 - x)**2 + (30 - y)**2) + 1.0
    )

    return (val,)


# Operators
toolbox.register("evaluate", eval_function)

# Two-point crossover
toolbox.register("mate", tools.cxTwoPoint)

# Flip-bit mutation
toolbox.register("mutate",
                 tools.mutFlipBit,
                 indpb=0.05)

# Tournament selection
toolbox.register("select",
                 tools.selTournament,
                 tournsize=3)


# Statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


def main(seed=0):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)

    CXPB, MUTPB = 0.5, 0.2

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        CXPB,
        MUTPB,
        100,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    print("-- End of evolution --")

    best_ind = hof[0]

    # Decode the best solution for displaying it
    x = decode(best_ind[0:30])
    y = decode(best_ind[30:60])
    z = decode(best_ind[60:90])

    print("Best individual (bits):", best_ind)
    print("Decoded values:")
    print("x =", x)
    print("y =", y)
    print("z =", z)
    print("Fitness =", best_ind.fitness.values)

    return pop, log, hof


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pop, log, hof = main(int(sys.argv[1]))
    else:
        pop, log, hof = main()