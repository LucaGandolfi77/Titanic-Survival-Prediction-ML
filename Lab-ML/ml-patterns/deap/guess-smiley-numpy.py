import random
from pathlib import Path

import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

POP_SIZE = 300
NGEN = 200
CXPB = 0.5
MUTPB = 0.2
TOURNSIZE = 3
BIT_MUTATION_PROB = 0.05
SMILEY_PATH = Path(__file__).with_name("smiley.txt")


def load_pattern(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = int(lines[0])
    cols = int(lines[1])
    grid = [[int(value) for value in line.split()] for line in lines[2:2 + rows]]
    flat = numpy.array([value for row in grid for value in row], dtype=int)
    return rows, cols, grid, flat


ROWS, COLS, TARGET_GRID, TARGET_PATTERN = load_pattern(SMILEY_PATH)
DIM = int(TARGET_PATTERN.size)


if not hasattr(creator, "FitnessMaxSmileyNumpy"):
    creator.create("FitnessMaxSmileyNumpy", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualSmileyNumpy"):
    creator.create("IndividualSmileyNumpy", numpy.ndarray, fitness=creator.FitnessMaxSmileyNumpy)


toolbox = base.Toolbox()
toolbox.register("attr_bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.IndividualSmileyNumpy, toolbox.attr_bit, n=DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    matches = int(numpy.sum(individual == TARGET_PATTERN))
    return (matches,)


def cx_two_point_copy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2].copy(),
        ind1[cxpoint1:cxpoint2].copy(),
    )
    return ind1, ind2


def format_pattern(flat_pattern):
    rows = []
    values = flat_pattern.tolist()
    for start in range(0, DIM, COLS):
        row = values[start:start + COLS]
        rows.append(" ".join(str(value) for value in row))
    return "\n".join(rows)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", cx_two_point_copy)
toolbox.register("mutate", tools.mutFlipBit, indpb=BIT_MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)


def main(seed=42):
    random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]
    best_matches = int(best.fitness.values[0])

    print("\n-- End of evolution --")
    print(f"Target size: {ROWS} x {COLS} = {DIM} bits")
    print(f"Best fitness (matching bits): {best_matches}/{DIM}")
    print("\nBest guessed pattern:")
    print(format_pattern(best))
    print("\nTarget pattern:")
    print(format_pattern(TARGET_PATTERN))

    return pop, logbook, hof


if __name__ == "__main__":
    main()