import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Read smiley.txt

with open("smiley.txt", "r") as f:
    rows = int(f.readline())
    cols = int(f.readline())

    target = []
    for line in f:
        target.extend(map(int, line.split()))

# DEAP setup

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    rows * cols
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

# Fitness function

def eval_function(individual):
    fitness = sum(
        individual[i] == target[i]
        for i in range(len(target))
    )

    return (fitness,)


toolbox.register("evaluate", eval_function)
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=0.05
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

# Statistics

stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# GA

def main(seed=0):

    random.seed(seed)

    pop = toolbox.population(n=300)

    hof = tools.HallOfFame(1)

    CXPB = 0.5
    MUTPB = 0.2

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

    print("Best fitness:", best_ind.fitness.values[0])

    print("\nBest individual:")

    for i in range(rows):
        print(
            " ".join(
                map(str, best_ind[i * cols:(i + 1) * cols])
            )
        )
    
    # Plot convergence
    
    gen = log.select("gen")
    avg = log.select("avg")
    best = log.select("max")

    plt.plot(gen, avg, label="Average Fitness")
    plt.plot(gen, best, label="Best Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Smiley convergence")
    plt.legend()
    plt.grid()
    plt.show()

    return pop, log, hof

if __name__ == "__main__":
    main()