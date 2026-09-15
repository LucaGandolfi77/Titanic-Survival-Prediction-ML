import random
import operator
import math
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from deap import base, creator, tools, algorithms, gp


random.seed(42)
np.random.seed(42)

SAMPLES = 3000
POPULATION = 200
GENERATIONS = 40
MAX_DEPTH = 6
K = 30
N_ERC = 5


def load_image(filename):
    return np.array(
        Image.open(filename).convert("L"),
        dtype=np.float32
    )


I = load_image("low_quality.jpg")
O = load_image("enhanced.jpg")

I_test = load_image("test_low.jpg")
O_test = load_image("test_enhanced.jpg")

if I.shape != O.shape:
    raise ValueError("low_quality.jpg and enhanced.jpg must have the same size")

if I_test.shape != O_test.shape:
    raise ValueError(
        "test_low.jpg and test_enhanced.jpg must have the same size"
    )


def clip(x):
    return min(255.0, max(0.0, x))


def add(a, b):
    return clip(a + b)


def sub(a, b):
    return clip(a - b)


def mul(a, b):
    return clip(a * b)


def div(a, b):
    if abs(b) < 0.000001:
        return clip(a)
    return clip(a / b)


def avg(a, b):
    return clip((a + b) / 2.0)


def minimum(a, b):
    return clip(min(a, b))


def maximum(a, b):
    return clip(max(a, b))


def median3(a, b, c):
    return clip(sorted([a, b, c])[1])


def contrast(a, b, c):
    return clip(a + b - c)


pset = gp.PrimitiveSet("FILTER", 9)

pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(avg, 2)
pset.addPrimitive(minimum, 2)
pset.addPrimitive(maximum, 2)
pset.addPrimitive(median3, 3)
pset.addPrimitive(contrast, 3)

for i in range(N_ERC):
    pset.addEphemeralConstant(
        f"ERC{i}",
        partial(random.uniform, 0, K)
    )

names = [
    "I_xm1_ym1",
    "I_xm1_y",
    "I_xm1_yp1",
    "I_x_ym1",
    "I_x_y",
    "I_x_yp1",
    "I_xp1_ym1",
    "I_xp1_y",
    "I_xp1_yp1"
]

for i, name in enumerate(names):
    pset.renameArguments(**{f"ARG{i}": name})


if not hasattr(creator, "ImageFitnessMin"):
    creator.create(
        "ImageFitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "ImageIndividual"):
    creator.create(
        "ImageIndividual",
        gp.PrimitiveTree,
        fitness=creator.ImageFitnessMin
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
    creator.ImageIndividual,
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


samples = []

for y in range(1, I.shape[0] - 1):
    for x in range(1, I.shape[1] - 1):
        neighborhood = I[
            y - 1:y + 2,
            x - 1:x + 2
        ].flatten()

        samples.append((neighborhood, O[y, x]))

random.shuffle(samples)
samples = samples[:SAMPLES]


def evaluate(individual):
    function = toolbox.compile(expr=individual)
    error = 0.0

    for neighborhood, expected in samples:
        try:
            predicted = function(*neighborhood)

            if not math.isfinite(predicted):
                return (1e20,)

            error += (clip(predicted) - expected) ** 2

        except Exception:
            return (1e20,)

    return (error / len(samples),)


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)

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

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=MAX_DEPTH
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=MAX_DEPTH
    )
)


population = toolbox.population(n=POPULATION)
hof = tools.HallOfFame(1)

stats = tools.Statistics(
    lambda individual: individual.fitness.values[0]
)

stats.register("avg", np.mean)
stats.register("min", np.min)

population, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.2,
    ngen=GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True
)


best_tree = hof[0]
best_function = toolbox.compile(expr=best_tree)


def apply_filter(image, function):
    result = image.copy()

    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            neighborhood = image[
                y - 1:y + 2,
                x - 1:x + 2
            ].flatten()

            try:
                value = function(*neighborhood)

                if math.isfinite(value):
                    result[y, x] = clip(value)

            except Exception:
                result[y, x] = image[y, x]

    return np.clip(result, 0, 255)


result_train = apply_filter(I, best_function)
result_test = apply_filter(I_test, best_function)

train_initial_mse = np.mean((I - O) ** 2)
train_final_mse = np.mean((result_train - O) ** 2)

test_initial_mse = np.mean((I_test - O_test) ** 2)
test_final_mse = np.mean((result_test - O_test) ** 2)

train_improvement = (
    100 * (train_initial_mse - train_final_mse)
    / train_initial_mse
)

test_improvement = (
    100 * (test_initial_mse - test_final_mse)
    / test_initial_mse
)

print("\nBest tree:")
print(best_tree)

print("\nTree depth:", best_tree.height)
print("Tree size:", len(best_tree))

print("\nTraining image")
print("Initial MSE:", train_initial_mse)
print("GP result MSE:", train_final_mse)
print("Improvement:", train_improvement, "%")

print("\nTest image")
print("Initial MSE:", test_initial_mse)
print("GP result MSE:", test_final_mse)
print("Improvement:", test_improvement, "%")


Image.fromarray(
    result_train.astype(np.uint8)
).save("gp_training_result.png")

Image.fromarray(
    result_test.astype(np.uint8)
).save("gp_test_result.png")


fig, axes = plt.subplots(2, 3, figsize=(13, 8))

images = [
    I,
    O,
    result_train,
    I_test,
    O_test,
    result_test
]

titles = [
    "Training input I",
    "Training target O",
    f"GP training result\nMSE = {train_final_mse:.2f}",
    "Test input",
    "Test target",
    f"GP test result\nMSE = {test_final_mse:.2f}"
]

for axis, image, title in zip(axes.flat, images, titles):
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

plt.tight_layout()
plt.savefig("gp_image_comparison.png", dpi=150)
plt.show()


generations = log.select("gen")
minimum = log.select("min")
average = log.select("avg")

plt.plot(generations, minimum, label="Best MSE")
plt.plot(generations, average, label="Average MSE")
plt.xlabel("Generation")
plt.ylabel("MSE")
plt.title("GP convergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("gp_image_convergence.png", dpi=150)
plt.show()