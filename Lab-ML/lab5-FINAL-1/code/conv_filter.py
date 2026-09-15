import random
import operator
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from deap import base, creator, tools, algorithms, gp


random.seed(42)
np.random.seed(42)

I = np.array(
    Image.open("greyscale.jpg").convert("L"),
    dtype=np.float32
)

noise = np.random.randint(-15, 16, size=I.shape)
O = np.clip(I + noise, 0, 255)

Image.fromarray(O.astype(np.uint8)).save("greyscale_noisy_02.png")


def protected_div(a, b):
    if abs(b) < 0.000001:
        return a
    return a / b


def average(a, b):
    return (a + b) / 2


def median3(a, b, c):
    return sorted([a, b, c])[1]


pset = gp.PrimitiveSet("FILTER", 9)

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(average, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(median3, 3)
pset.addPrimitive(operator.neg, 1)

for i in range(9):
    pset.renameArguments(**{f"ARG{i}": f"p{i}"})


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


samples = []

for y in range(1, I.shape[0] - 1):
    for x in range(1, I.shape[1] - 1):

        neighborhood = O[
            y - 1:y + 2,
            x - 1:x + 2
        ].flatten()

        samples.append(
            (neighborhood, I[y, x])
        )


random.shuffle(samples)
samples = samples[:3000]


def evaluate(individual):

    function = toolbox.compile(expr=individual)
    error = 0

    for neighborhood, expected in samples:

        try:
            predicted = function(*neighborhood)

            if not math.isfinite(predicted):
                return (1e20,)

            predicted = np.clip(predicted, 0, 255)
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
        max_value=8
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)


population = toolbox.population(n=150)
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
    ngen=30,
    stats=stats,
    halloffame=hof,
    verbose=True
)


best = hof[0]
filter_function = toolbox.compile(expr=best)

print("\nBest filter:")
print(best)

print("\nBest training MSE:")
print(best.fitness.values[0])


filtered = O.copy()

for y in range(1, O.shape[0] - 1):
    for x in range(1, O.shape[1] - 1):

        neighborhood = O[
            y - 1:y + 2,
            x - 1:x + 2
        ].flatten()

        try:
            value = filter_function(*neighborhood)

            if math.isfinite(value):
                filtered[y, x] = np.clip(value, 0, 255)

        except Exception:
            filtered[y, x] = O[y, x]


filtered = np.clip(filtered, 0, 255)

Image.fromarray(
    filtered.astype(np.uint8)
).save("greyscale_filtered.png")


noisy_mse = np.mean((I - O) ** 2)
filtered_mse = np.mean((I - filtered) ** 2)

print("\nNoisy image MSE:", noisy_mse)
print("Filtered image MSE:", filtered_mse)

improvement = 100 * (
    noisy_mse - filtered_mse
) / noisy_mse

print("Improvement:", improvement, "%")


fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(I, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(O, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(f"Noisy\nMSE = {noisy_mse:.2f}")
axes[1].axis("off")

axes[2].imshow(filtered, cmap="gray", vmin=0, vmax=255)
axes[2].set_title(f"Filtered\nMSE = {filtered_mse:.2f}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("greyscale_comparison.png", dpi=150)
plt.show()