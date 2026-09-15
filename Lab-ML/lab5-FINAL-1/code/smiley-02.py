import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Read smiley.txt

with open("smiley_large_20.txt", "r") as f:
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
        verbose=False
    )

    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    print(f"Seed {seed} - Best fitness: {best_fitness}")

    best_per_generation = log.select("max")

    avg_per_generation = log.select("avg")

    return best_fitness, best_ind, best_per_generation, avg_per_generation

# Execution 5 times

if __name__ == "__main__":

    number_of_runs = 5

    fitness_results = []
    best_individuals = []
    all_best_curves = []
    all_average_curves = []

    for run in range(number_of_runs):

        print(f"\nRun {run + 1}/{number_of_runs}")

        best_fitness, best_ind, best_curve, average_curve = main(
            seed=run
        )

        fitness_results.append(best_fitness)
        best_individuals.append(best_ind)
        all_best_curves.append(best_curve)
        all_average_curves.append(average_curve)

    mean_fitness = np.mean(fitness_results)
    std_fitness = np.std(fitness_results)

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    print("Fitness values:", fitness_results)
    print("Mean fitness:", mean_fitness)
    print("Standard deviation:", std_fitness)
    print("Minimum fitness:", np.min(fitness_results))
    print("Maximum fitness:", np.max(fitness_results))

    best_run_index = int(np.argmax(fitness_results))
    overall_best = best_individuals[best_run_index]

    print("\nBest run:", best_run_index + 1)
    print("Overall best fitness:", fitness_results[best_run_index])

    #print("\nOverall best individual:")

    #for i in range(rows):
#
    #    print(
    #        " ".join(
    #            map(
    #                str,
    #                overall_best[i * cols:(i + 1) * cols]
    #            )
    #        )
    #    )

    mean_best_curve = np.mean(
        np.array(all_best_curves),
        axis=0
    )

    mean_average_curve = np.mean(
        np.array(all_average_curves),
        axis=0
    )

    generations = range(len(mean_best_curve))

    plt.plot(
        generations,
        mean_average_curve,
        label="Mean Population Fitness"
    )

    plt.plot(
        generations,
        mean_best_curve,
        label="Mean Best Fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Average convergence over 5 runs")
    plt.legend()
    plt.grid()

    plt.show()