import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

LOW, HIGH = -250.0, 250.0
DIM = 3
BITS_PER_VAR = 30
TOTAL_BITS = DIM * BITS_PER_VAR
BIT_MAX = (1 << BITS_PER_VAR) - 1
POP_SIZE = 300
NGEN = 300
CXPB = 0.5
MUTPB = 0.2
RUNS = 5
PLOT_PATH = "convergence-comparison.png"


if not hasattr(creator, "FitnessMinFloat"):
    creator.create("FitnessMinFloat", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualFloat"):
    creator.create("IndividualFloat", list, fitness=creator.FitnessMinFloat)
if not hasattr(creator, "FitnessMinBits"):
    creator.create("FitnessMinBits", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualBits"):
    creator.create("IndividualBits", list, fitness=creator.FitnessMinBits)


def objective(x, y, z):
    return (1.5 + math.sin(z)) * (math.sqrt((20 - x) ** 2 + (30 - y) ** 2) + 1)


def clamp_float(individual):
    for i in range(len(individual)):
        individual[i] = min(HIGH, max(LOW, individual[i]))
    return individual


def decode_chunk(bits):
    integer_value = 0
    for bit in bits:
        integer_value = (integer_value << 1) | int(bit)
    # Value(i) = Min + i / (2^N - 1) * (Max - Min)
    return LOW + (HIGH - LOW) * integer_value / BIT_MAX


def decode_bits_individual(individual):
    return tuple(
        decode_chunk(individual[index:index + BITS_PER_VAR])
        for index in range(0, TOTAL_BITS, BITS_PER_VAR)
    )


def evaluate_float(individual):
    x, y, z = individual
    return (objective(x, y, z),)


def evaluate_bits(individual):
    x, y, z = decode_bits_individual(individual)
    return (objective(x, y, z),)


def mate_float(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    clamp_float(ind1)
    clamp_float(ind2)
    return ind1, ind2


def mutate_float(individual):
    tools.mutGaussian(individual, mu=0.0, sigma=0.2, indpb=1.0 / DIM)
    clamp_float(individual)
    return (individual,)


float_toolbox = base.Toolbox()
float_toolbox.register("attr_real", random.uniform, LOW, HIGH)
float_toolbox.register("individual", tools.initRepeat, creator.IndividualFloat, float_toolbox.attr_real, DIM)
float_toolbox.register("population", tools.initRepeat, list, float_toolbox.individual)
float_toolbox.register("evaluate", evaluate_float)
float_toolbox.register("mate", mate_float)
float_toolbox.register("mutate", mutate_float)
float_toolbox.register("select", tools.selTournament, tournsize=3)


bit_toolbox = base.Toolbox()
bit_toolbox.register("attr_bit", random.randint, 0, 1)
bit_toolbox.register("individual", tools.initRepeat, creator.IndividualBits, bit_toolbox.attr_bit, TOTAL_BITS)
bit_toolbox.register("population", tools.initRepeat, list, bit_toolbox.individual)
bit_toolbox.register("evaluate", evaluate_bits)
bit_toolbox.register("mate", tools.cxTwoPoint)
bit_toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / TOTAL_BITS)
bit_toolbox.register("select", tools.selTournament, tournsize=3)


def run_ga(toolbox, seed, decode_individual, verbose=False):
    random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    best = hof[0]
    x, y, z = decode_individual(best)
    return {
        "population": pop,
        "log": log,
        "hof": hof,
        "best_individual": best,
        "decoded": (x, y, z),
        "fitness": best.fitness.values[0],
    }


def identity_decode(individual):
    return tuple(individual)


def compare_versions():
    float_results = []
    bit_results = []

    for seed in range(RUNS):
        float_results.append(run_ga(float_toolbox, seed, identity_decode, verbose=False))
        bit_results.append(run_ga(bit_toolbox, seed, decode_bits_individual, verbose=False))

    best_float = min(float_results, key=lambda result: result["fitness"])
    best_bits = min(bit_results, key=lambda result: result["fitness"])
    plot_convergence(float_results, bit_results, PLOT_PATH)

    print("Function minimization with DEAP")
    print("f(x,y,z) = (1.5 + sin(z)) * (sqrt((20 - x)^2 + (30 - y)^2) + 1)")
    print(f"Domain: x, y, z in [{LOW}, {HIGH}]")
    print()
    print("Version 1: list of 3 floating-point numbers")
    print("- representation: [x, y, z]")
    print("- crossover: two-point crossover")
    print("- mutation: Gaussian, mu=0, sigma=0.2")
    print()
    print("Version 2: list of 90 bits")
    print("- representation: 30 bits for each variable x, y, z")
    print("- crossover: two-point crossover")
    print("- mutation: flip-bit mutation")
    print()
    print("Best result over", RUNS, "runs")
    print("-" * 72)

    for label, result_set in (("Float", best_float), ("Bits ", best_bits)):
        x, y, z = result_set["decoded"]
        print(
            f"{label} -> fitness={result_set['fitness']:.12f} | "
            f"x={x:.6f}, y={y:.6f}, z={z:.6f}"
        )

    print("-" * 72)
    print(
        f"Average best fitness over {RUNS} runs: "
        f"float={numpy.mean([r['fitness'] for r in float_results]):.12f}, "
        f"bits={numpy.mean([r['fitness'] for r in bit_results]):.12f}"
    )
    print(f"Convergence plot saved to: {PLOT_PATH}")

    return float_results, bit_results


def aggregate_logbooks(results):
    generations = results[0]["log"].select("gen")
    avg_series = numpy.mean([result["log"].select("avg") for result in results], axis=0)
    min_series = numpy.mean([result["log"].select("min") for result in results], axis=0)
    return generations, avg_series, min_series


def plot_convergence(float_results, bit_results, output_path):
    float_gens, float_avg, float_min = aggregate_logbooks(float_results)
    bit_gens, bit_avg, bit_min = aggregate_logbooks(bit_results)

    plt.figure(figsize=(11, 6))
    plt.plot(float_gens, float_avg, label="Float avg fitness", color="#2563eb", linewidth=2)
    plt.plot(float_gens, float_min, label="Float best fitness", color="#1d4ed8", linestyle="--", linewidth=2)
    plt.plot(bit_gens, bit_avg, label="Bits avg fitness", color="#dc2626", linewidth=2)
    plt.plot(bit_gens, bit_min, label="Bits best fitness", color="#991b1b", linestyle="--", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("DEAP convergence comparison: float vs bit representation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    compare_versions()
