import operator
import math
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):

    if abs(right) < 0.000001:
        return 1.0

    return left / right

def target_function(x, y):
    """
    Function to approximate.

    The exponential function is intentionally NOT included
    in the GP function set, so GP cannot reproduce this
    expression exactly.
    """

    return math.exp(-(x**2 + y**2)) + 0.25 * x * y

# Two input arguments: x and y
pset = gp.PrimitiveSet("MAIN", 2)

# Function set F
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# Terminal set:
# T = {x, y, ERC}
#
# ERC is an Ephemeral Random Constant in [1, 10].
pset.addEphemeralConstant(
    "ERC",
    partial(random.uniform, 1.0, 10.0)
)

pset.renameArguments(
    ARG0="x",
    ARG1="y"
)

# The hasattr checks avoid errors when the file is executed
# multiple times in a notebook.
if not hasattr(creator, "FitnessMin"):
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        gp.PrimitiveTree,
        fitness=creator.FitnessMin
    )

toolbox = base.Toolbox()

toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=1,
    max_=3
)

toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.expr
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register(
    "compile",
    gp.compile,
    pset=pset
)

# 25 x 25 = 625 training points.
training_axis = np.linspace(-2.0, 2.0, 25)

training_points = [
    (x, y)
    for x in training_axis
    for y in training_axis
]

def evalSymbReg(individual, points):
    """
    Compile the GP tree and calculate the mean squared error
    over all training points.
    """

    function = toolbox.compile(expr=individual)

    squared_errors = []

    for x, y in points:
        try:
            predicted = function(x, y)
            expected = target_function(x, y)

            # Penalize invalid numerical results.
            if not math.isfinite(predicted):
                return (1e10,)

            error = (predicted - expected) ** 2
            squared_errors.append(error)

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError,
            TypeError
        ):
            return (1e10,)

    mse = math.fsum(squared_errors) / len(points)

    return (mse,)


toolbox.register(
    "evaluate",
    evalSymbReg,
    points=training_points
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

toolbox.register(
    "mate",
    gp.cxOnePoint
)

toolbox.register(
    "expr_mut",
    gp.genFull,
    min_=0,
    max_=2
)

toolbox.register(
    "mutate",
    gp.mutUniform,
    expr=toolbox.expr_mut,
    pset=pset
)


# Avoid excessively large trees.
toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=17
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=17
    )
)

def run_experiment(
    name,
    population_size,
    generations,
    crossover_probability,
    mutation_probability,
    seed
):
    """Run one GP configuration."""

    random.seed(seed)
    np.random.seed(seed)

    population = toolbox.population(
        n=population_size
    )

    hall_of_fame = tools.HallOfFame(1)

    fitness_statistics = tools.Statistics(
        lambda individual: individual.fitness.values
    )

    size_statistics = tools.Statistics(len)

    statistics = tools.MultiStatistics(
        fitness=fitness_statistics,
        size=size_statistics
    )

    statistics.register("avg", np.mean)
    statistics.register("std", np.std)
    statistics.register("min", np.min)
    statistics.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        ngen=generations,
        stats=statistics,
        halloffame=hall_of_fame,
        verbose=False
    )

    best_individual = hall_of_fame[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Population:", population_size)
    print("Generations:", generations)
    print("Crossover probability:", crossover_probability)
    print("Mutation probability:", mutation_probability)
    print("Best expression:", best_individual)
    print("Best MSE:", best_fitness)
    print("Tree size:", len(best_individual))
    print("Tree height:", best_individual.height)

    return {
        "name": name,
        "population": population,
        "logbook": logbook,
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "population_size": population_size,
        "generations": generations,
        "crossover_probability": crossover_probability,
        "mutation_probability": mutation_probability
    }

def evaluate_on_grid(individual, x_grid, y_grid):
    """Evaluate the evolved expression over a 2D grid."""

    function = toolbox.compile(expr=individual)

    z_grid = np.zeros_like(
        x_grid,
        dtype=float
    )

    for row in range(x_grid.shape[0]):
        for column in range(x_grid.shape[1]):

            x = x_grid[row, column]
            y = y_grid[row, column]

            try:
                value = function(x, y)

                if math.isfinite(value):
                    z_grid[row, column] = value
                else:
                    z_grid[row, column] = np.nan

            except Exception:
                z_grid[row, column] = np.nan

    return z_grid

def plot_results(results):
    """Plot the target function and all evolved approximations."""

    # Dense sampling for visualization.
    dense_axis = np.linspace(-2.0, 2.0, 80)

    x_grid, y_grid = np.meshgrid(
        dense_axis,
        dense_axis
    )

    z_target = np.exp(
        -(x_grid**2 + y_grid**2)
    ) + 0.25 * x_grid * y_grid

    number_of_plots = len(results) + 1

    figure = plt.figure(
        figsize=(5 * number_of_plots, 5)
    )

    # Original function
    axis = figure.add_subplot(
        1,
        number_of_plots,
        1,
        projection="3d"
    )

    axis.plot_surface(
        x_grid,
        y_grid,
        z_target,
        cmap="viridis"
    )

    axis.set_title("Original function")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("f(x, y)")

    # GP approximations
    for index, result in enumerate(results, start=2):

        z_approximation = evaluate_on_grid(
            result["best_individual"],
            x_grid,
            y_grid
        )

        axis = figure.add_subplot(
            1,
            number_of_plots,
            index,
            projection="3d"
        )

        axis.plot_surface(
            x_grid,
            y_grid,
            z_approximation,
            cmap="plasma"
        )

        axis.set_title(
            result["name"]
            + "\nMSE = "
            + f"{result['best_fitness']:.6f}"
        )

        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("Approximation")

    plt.tight_layout()
    plt.savefig(
        "gp_2d_comparison.png",
        dpi=150
    )
    plt.show()

def plot_convergence(results):
    """Compare the fitness convergence of the experiments."""

    for result in results:

        generations = result["logbook"].select("gen")

        minimum_fitness = result[
            "logbook"
        ].chapters["fitness"].select("min")

        plt.plot(
            generations,
            minimum_fitness,
            label=result["name"]
        )

    plt.xlabel("Generation")
    plt.ylabel("Minimum MSE")
    plt.title("GP parameter comparison")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "gp_convergence.png",
        dpi=150
    )

    plt.show()

def main():

    experiments = [
        {
            "name": "Small experiment",
            "population_size": 100,
            "generations": 20,
            "crossover_probability": 0.5,
            "mutation_probability": 0.1,
            "seed": 318
        },
        {
            "name": "Original-like experiment",
            "population_size": 300,
            "generations": 40,
            "crossover_probability": 0.5,
            "mutation_probability": 0.1,
            "seed": 318
        },
        {
            "name": "Larger experiment",
            "population_size": 500,
            "generations": 80,
            "crossover_probability": 0.7,
            "mutation_probability": 0.2,
            "seed": 318
        }
    ]

    results = []

    for parameters in experiments:
        result = run_experiment(**parameters)
        results.append(result)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for result in results:
        print(
            f"{result['name']:25} | "
            f"MSE = {result['best_fitness']:.8f} | "
            f"size = {len(result['best_individual']):3} | "
            f"height = {result['best_individual'].height:2}"
        )

    best_result = min(
        results,
        key=lambda result: result["best_fitness"]
    )

    print("\nBest experiment:", best_result["name"])
    print("Best expression:", best_result["best_individual"])
    print("Best MSE:", best_result["best_fitness"])

    plot_results(results)
    plot_convergence(results)

    return results


if __name__ == "__main__":
    results = main()