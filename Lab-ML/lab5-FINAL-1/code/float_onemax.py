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

# Attribute generator
toolbox.register("attr_float", random.uniform, -250, 250)

# Structure initializers
# Individual = list of 3 floating point numbers
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, 3)

# Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function
def eval_function(individual):
    x, y, z = individual

    val = (1.5 + np.sin(z)) * (
        np.sqrt((20 - x)**2 + (30 - y)**2) + 1.0
    )

    return (val,)


# Operators
toolbox.register("evaluate", eval_function)

# Two-point crossover
toolbox.register("mate", tools.cxTwoPoint)

# Gaussian mutation
toolbox.register("mutate", tools.mutGaussian,
                 mu=0,
                 sigma=0.2,
                 indpb=0.05)

# Tournament selection
toolbox.register("select", tools.selTournament, tournsize=3)


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

    print("Best individual is %s, %s" %
          (best_ind, best_ind.fitness.values))

    return pop, log, hof


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pop, log, hof = main(int(sys.argv[1]))
    else:
        pop, log, hof = main()