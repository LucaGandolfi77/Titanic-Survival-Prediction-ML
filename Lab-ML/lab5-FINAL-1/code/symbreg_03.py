
import operator
import math
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# ============================================================
# 1. Image utilities
# ============================================================

def load_image(filename):
    """
    Load a grayscale image and normalize pixels to [0, 1].
    """
    image = Image.open(filename).convert("L")
    array = np.asarray(image, dtype=np.float64) / 255.0
    return array


def save_image(array, filename):
    """
    Save a [0,1] image as an 8-bit grayscale image.
    """
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255).astype(np.uint8)

    image = Image.fromarray(array, mode="L")
    image.save(filename)


# ============================================================
# 2. Protected / image operators
# ============================================================

def protectedDiv(left, right):
    if abs(right) < 0.000001:
        return 1.0
    return left / right


def protectedSqrt(value):
    return math.sqrt(abs(value))


def protectedLog(value):
    return math.log(abs(value) + 0.000001)


def protectedExp(value):
    """
    Avoid excessively large exponential values.
    """
    value = max(-10.0, min(10.0, value))
    return math.exp(value)


def clip(value):
    return max(0.0, min(1.0, value))


# ============================================================
# 3. Read images
# ============================================================

# I(x,y): low-quality input image
input_image = load_image("greyscale_bw_burned.png")

# O(x,y): manually enhanced target image
output_image = load_image("greyscale_bw_enhanced.png")

# Second image used ONLY for testing generality
test_image = load_image("greyscale.png")

if input_image.shape != output_image.shape:
    raise ValueError(
        "input.png and output.png must have the same dimensions"
    )


# ============================================================
# 4. GP Primitive Set
# ============================================================

# Three inputs:
#
# I = intensity of the input image
# x = horizontal coordinate
# y = vertical coordinate

pset = gp.PrimitiveSet("MAIN", 3)

# Basic arithmetic
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)

# Unary operators
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(abs, 1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(protectedExp, 1)

# Trigonometric operators
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# Useful image operators
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(clip, 1)


# Random constants
pset.addEphemeralConstant(
    "ERC",
    partial(random.uniform, -1.0, 1.0)
)


# Rename arguments
pset.renameArguments(
    ARG0="I",
    ARG1="x",
    ARG2="y"
)


# ============================================================
# 5. DEAP classes
# ============================================================

if not hasattr(creator, "FitnessMinImage"):
    creator.create(
        "FitnessMinImage",
        base.Fitness,
        weights=(-1.0,)
    )


if not hasattr(creator, "IndividualImage"):
    creator.create(
        "IndividualImage",
        gp.PrimitiveTree,
        fitness=creator.FitnessMinImage
    )


toolbox = base.Toolbox()


# ============================================================
# 6. Individuals
# ============================================================

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
    creator.IndividualImage,
    toolbox.expr
)


toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)


# Compile GP tree
toolbox.register(
    "compile",
    gp.compile,
    pset=pset
)


# ============================================================
# 7. Training pixels
# ============================================================

height, width = input_image.shape

training_points = []

for y in range(height):
    for x in range(width):

        training_points.append(
            (
                input_image[y, x],
                x / max(1, width - 1),
                y / max(1, height - 1),
                output_image[y, x]
            )
        )


# ============================================================
# 8. Fitness function
# ============================================================

fitness_calls = 0


def evalImage(individual, points):
    """
    Evaluate the GP transformation.

    The GP receives:
        I = input pixel
        x = normalized x coordinate
        y = normalized y coordinate

    and must produce the corresponding output pixel.
    """

    global fitness_calls
    fitness_calls += 1

    function = toolbox.compile(expr=individual)

    squared_errors = []

    for I, x, y, expected in points:

        try:

            predicted = function(I, x, y)

            if not math.isfinite(predicted):
                return (1e10,)

            # Keep the result in a valid grayscale range
            predicted = clip(predicted)

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
    evalImage,
    points=training_points
)


# ============================================================
# 9. GA operators
# ============================================================

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


# Prevent excessively large trees

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=10
    )
)


toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=10
    )
)


# ============================================================
# 10. Apply GP tree to an image
# ============================================================

def apply_tree(individual, image):
    """
    Apply the evolved GP transformation to every pixel.
    """

    function = toolbox.compile(expr=individual)

    height, width = image.shape

    result = np.zeros_like(
        image,
        dtype=float
    )

    for y in range(height):
        for x in range(width):

            I = image[y, x]

            normalized_x = x / max(1, width - 1)
            normalized_y = y / max(1, height - 1)

            try:

                value = function(
                    I,
                    normalized_x,
                    normalized_y
                )

                if math.isfinite(value):
                    result[y, x] = clip(value)
                else:
                    result[y, x] = 0.0

            except Exception:
                result[y, x] = 0.0

    return result


# ============================================================
# 11. Main GP experiment
# ============================================================

def main():

    global fitness_calls

    random.seed(318)
    np.random.seed(318)

    fitness_calls = 0

    population_size = 300
    generations = 50

    crossover_probability = 0.5
    mutation_probability = 0.1

    population = toolbox.population(
        n=population_size
    )

    hall_of_fame = tools.HallOfFame(1)


    # Statistics

    fitness_statistics = tools.Statistics(
        lambda individual: individual.fitness.values
    )

    size_statistics = tools.Statistics(len)

    statistics = tools.MultiStatistics(
        fitness=fitness_statistics,
        size=size_statistics
    )

    statistics.register(
        "avg",
        np.mean
    )

    statistics.register(
        "std",
        np.std
    )

    statistics.register(
        "min",
        np.min
    )

    statistics.register(
        "max",
        np.max
    )


    # ========================================================
    # Evolution
    # ========================================================

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        ngen=generations,
        stats=statistics,
        halloffame=hall_of_fame,
        verbose=True
    )


    # ========================================================
    # Best solution
    # ========================================================

    best_individual = hall_of_fame[0]

    best_fitness = (
        best_individual.fitness.values[0]
    )


    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("Best GP tree:")
    print(best_individual)

    print("\nBest MSE:")
    print(best_fitness)

    print("\nTree size:")
    print(len(best_individual))

    print("\nTree height:")
    print(best_individual.height)

    print("\nFitness function calls:")
    print(fitness_calls)


    # ========================================================
    # 12. Apply tree to training image
    # ========================================================

    generated_image = apply_tree(
        best_individual,
        input_image
    )

    save_image(
        generated_image,
        "gp_result.png"
    )


    # ========================================================
    # 13. Generality test
    # ========================================================

    test_result = apply_tree(
        best_individual,
        test_image
    )

    save_image(
        test_result,
        "gp_test_result.png"
    )


    # ========================================================
    # 14. Calculate test MSE
    # ========================================================

    # If a manually enhanced version of the test image exists,
    # it can be compared here.
    #
    # For now we only save the result.


    # ========================================================
    # 15. Plot images
    # ========================================================

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        input_image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Input I(x,y)")

    axes[1].imshow(
        output_image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Target O(x,y)")

    axes[2].imshow(
        generated_image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title("GP result")

    axes[3].imshow(
        test_result,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].set_title("GP on unseen image")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        "gp_image_results.png",
        dpi=150
    )

    plt.show()


    # ========================================================
    # 16. Convergence graph
    # ========================================================

    generations_log = (
        logbook.select("gen")
    )

    minimum_fitness = (
        logbook
        .chapters["fitness"]
        .select("min")
    )

    average_fitness = (
        logbook
        .chapters["fitness"]
        .select("avg")
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        generations_log,
        minimum_fitness,
        label="Best / minimum MSE"
    )

    plt.plot(
        generations_log,
        average_fitness,
        label="Average MSE"
    )

    plt.xlabel("Generation")
    plt.ylabel("MSE")

    plt.title(
        "GP image enhancement convergence"
    )

    plt.yscale("log")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "gp_image_convergence.png",
        dpi=150
    )

    plt.show()


    return (
        population,
        logbook,
        hall_of_fame
    )


if __name__ == "__main__":
    main()
