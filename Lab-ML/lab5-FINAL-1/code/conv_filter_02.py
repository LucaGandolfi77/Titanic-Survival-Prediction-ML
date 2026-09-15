import random
import operator
import math
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from deap import base, creator, tools, algorithms, gp


TRAIN_IMAGE = "greyscale.jpg"
TEST_IMAGE = "greyscale_noisy_01.jpg"

TRAIN_NOISE = 15
TEST_NOISE = 30

N_ERC = 5
K = 30
MAX_DEPTH = 6

random.seed(42)
np.random.seed(42)


def clip(value):
    return min(255.0, max(0.0, value))


def add(a, b):
    return clip(a + b)


def sub(a, b):
    return clip(a - b)


def mul(a, b):
    return clip(a * b)


def protected_div(a, b):
    if abs(b) < 0.000001:
        return clip(a)
    return clip(a / b)


def average(a, b):
    return clip((a + b) / 2.0)


def minimum(a, b):
    return clip(min(a, b))


def maximum(a, b):
    return clip(max(a, b))


def median3(a, b, c):
    return clip(sorted([a, b, c])[1])


def load_image(filename):
    return np.array(
        Image.open(filename).convert("L"),
        dtype=np.float32
    )


def add_noise(image, noise_level):
    noise = np.random.randint(
        -noise_level,
        noise_level + 1,
        size=image.shape
    )

    return np.clip(image + noise, 0, 255)


I_train = load_image(TRAIN_IMAGE)
O_train = add_noise(I_train, TRAIN_NOISE)

try:
    I_test = load_image(TEST_IMAGE)
    print("Testing on:", TEST_IMAGE)
except FileNotFoundError:
    I_test = I_train.copy()
    print("greyscale_test.jpg not found.")
    print("Testing on greyscale.jpg with different noise.")

O_test = add_noise(I_test, TEST_NOISE)


pset = gp.PrimitiveSet("FILTER", 9)

pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(average, 2)
pset.addPrimitive(minimum, 2)
pset.addPrimitive(maximum, 2)
pset.addPrimitive(median3, 3)

for i in range(N_ERC):
    pset.addEphemeralConstant(
        f"ERC{i}",
        partial(random.uniform, 0, K)
    )


names = [
    "O_xm1_ym1",
    "O_xm1_y",
    "O_xm1_yp1",
    "O_x_ym1",
    "O_x_y",
    "O_x_yp1",
    "O_xp1_ym1",
    "O_xp1_y",
    "O_xp1_yp1"
]

for i, name in enumerate(names):
    pset.renameArguments(**{f"ARG{i}": name})


if not hasattr(creator, "DenoisingFitnessMin"):
    creator.create(
        "DenoisingFitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "DenoisingIndividual"):
    creator.create(
        "DenoisingIndividual",
        gp.PrimitiveTree,
        fitness=creator.DenoisingFitnessMin
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
    creator.DenoisingIndividual,
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

for y in range(1, I_train.shape[0] - 1):
    for x in range(1, I_train.shape[1] - 1):

        neighborhood = O_train[
            y - 1:y + 2,
            x - 1:x + 2
        ].flatten()

        samples.append(
            (neighborhood, I_train[y, x])
        )


random.shuffle(samples)
samples = samples[:3000]


def evaluate(individual):
    function = toolbox.compile(expr=individual)
    error = 0.0

    for neighborhood, expected in samples:
        try:
            predicted = function(*neighborhood)

            if not math.isfinite(predicted):
                return (1e20,)

            predicted = clip(predicted)
            error += (predicted - expected) ** 2

        except Exception:
            return (1e20,)

    return (error / len(samples),)


toolbox.register("evaluate", evaluate)

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


population = toolbox.population(n=200)
hof = tools.HallOfFame(1)

stats = tools.Statistics(
    lambda individual: individual.fitness.values[0]
)

stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)


population, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.2,
    ngen=40,
    stats=stats,
    halloffame=hof,
    verbose=True
)


best_tree = hof[0]
best_filter = toolbox.compile(expr=best_tree)

print("\nBest filter:")
print(best_tree)

print("\nTree depth:", best_tree.height)
print("Tree size:", len(best_tree))
print("Training MSE:", best_tree.fitness.values[0])


def apply_filter(noisy_image, function):
    filtered = noisy_image.copy()

    for y in range(1, noisy_image.shape[0] - 1):
        for x in range(1, noisy_image.shape[1] - 1):

            neighborhood = noisy_image[
                y - 1:y + 2,
                x - 1:x + 2
            ].flatten()

            try:
                value = function(*neighborhood)

                if math.isfinite(value):
                    filtered[y, x] = clip(value)

            except Exception:
                filtered[y, x] = noisy_image[y, x]

    return np.clip(filtered, 0, 255)


filtered_train = apply_filter(O_train, best_filter)
filtered_test = apply_filter(O_test, best_filter)


train_noisy_mse = np.mean((I_train - O_train) ** 2)
train_filtered_mse = np.mean((I_train - filtered_train) ** 2)

test_noisy_mse = np.mean((I_test - O_test) ** 2)
test_filtered_mse = np.mean((I_test - filtered_test) ** 2)


train_improvement = 100 * (
    train_noisy_mse - train_filtered_mse
) / train_noisy_mse

test_improvement = 100 * (
    test_noisy_mse - test_filtered_mse
) / test_noisy_mse


print("\nTRAINING IMAGE")
print("Noise level:", TRAIN_NOISE)
print("Noisy MSE:", train_noisy_mse)
print("Filtered MSE:", train_filtered_mse)
print("Improvement:", train_improvement, "%")

print("\nTEST IMAGE")
print("Noise level:", TEST_NOISE)
print("Noisy MSE:", test_noisy_mse)
print("Filtered MSE:", test_filtered_mse)
print("Improvement:", test_improvement, "%")


Image.fromarray(
    O_train.astype(np.uint8)
).save("training_noisy.png")

Image.fromarray(
    filtered_train.astype(np.uint8)
).save("training_filtered.png")

Image.fromarray(
    O_test.astype(np.uint8)
).save("test_noisy.png")

Image.fromarray(
    filtered_test.astype(np.uint8)
).save("test_filtered.png")


fig, axes = plt.subplots(2, 3, figsize=(13, 8))

axes[0, 0].imshow(I_train, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Training original")

axes[0, 1].imshow(O_train, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title(
    f"Training noisy\nMSE = {train_noisy_mse:.2f}"
)

axes[0, 2].imshow(
    filtered_train,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 2].set_title(
    f"Training filtered\nMSE = {train_filtered_mse:.2f}"
)

axes[1, 0].imshow(I_test, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("Test original")

axes[1, 1].imshow(O_test, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title(
    f"Test noisy\nMSE = {test_noisy_mse:.2f}"
)

axes[1, 2].imshow(
    filtered_test,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 2].set_title(
    f"Test filtered\nMSE = {test_filtered_mse:.2f}"
)

for axis in axes.flat:
    axis.axis("off")

plt.tight_layout()
plt.savefig("denoising_comparison.png", dpi=150)
plt.show()


generations = log.select("gen")
minimum_fitness = log.select("min")
average_fitness = log.select("avg")

plt.plot(
    generations,
    minimum_fitness,
    label="Best MSE"
)

plt.plot(
    generations,
    average_fitness,
    label="Average MSE"
)

plt.xlabel("Generation")
plt.ylabel("MSE")
plt.title("GP filter convergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("denoising_convergence.png", dpi=150)
plt.show()