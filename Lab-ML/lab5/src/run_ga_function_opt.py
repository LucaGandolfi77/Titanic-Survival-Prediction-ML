"""Lab 5 - Evolutionary Computation: Genetic Algorithms with DEAP.

Covers the exercises on pages 51-72 of the merged deck.  This script implements
**Exercise 1 (function optimisation)**, which is fully specified by the deck and
needs no additional data files:

    minimise  f(x, y, z) = (1.5 + sin(z)) * ( sqrt((20 - x)^2 + (30 - y)^2) + 1 )
    for       x, y, z in [-250, 250]

Two representations are compared, exactly as the exercise requires:

* **A. list of 3 floats** - Gaussian mutation (mu=0, sigma=0.2) and two-point
  crossover.
* **B. list of 90 bits** - 30 bits per variable, decoded with the formula on
  p.65:  value(i) = Min + i/(2^N - 1) * (Max - Min); two-point crossover and
  flip-bit mutation.

The analysis required by the deck is a comparison of *convergence speed* using
the logbook, plotting best and average fitness against generation number.

Design note: in representation B the objective function takes the *float* values,
so the fitness function must decode the bit string first.  That decode step is
the extra work the deck refers to on p.65.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "lab1", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arff_utils import LAB1_DIR, write_csv  # noqa: E402

LAB5_DIR = os.path.join(LAB1_DIR, "..", "lab5")
LAB5_RESULTS = os.path.join(LAB5_DIR, "results")
LAB5_FIGURES = os.path.join(LAB5_DIR, "figures")

BOUNDS = (-250.0, 250.0)
N_BITS = 30                # bits per variable in representation B
N_VARS = 3
IND_LEN = N_BITS * N_VARS  # 90 bits

#: The analytic minimum.  f is a product of two non-negative factors:
#:   * (1.5 + sin z) is minimised at sin z = -1, giving 0.5;
#:   * (sqrt((20-x)^2+(30-y)^2) + 1) is minimised at (x, y) = (20, 30), giving 1.
#: Both can be satisfied simultaneously, so the global minimum is 0.5 * 1 = 0.5.
#: Note that sin z = -1 has 80 solutions in [-250, 250] (z = -pi/2 + 2k*pi), so the
#: landscape is strongly MULTIMODAL with 80 equivalent global optima - which is
#: precisely what makes this a non-trivial test for a GA.
ANALYTIC_ARGMIN = (20.0, 30.0, -np.pi / 2)
ANALYTIC_MIN = 0.5


def objective(x: float, y: float, z: float) -> float:
    """The function to minimise (p.63)."""
    return (1.5 + np.sin(z)) * (np.sqrt((20.0 - x) ** 2 + (30.0 - y) ** 2) + 1.0)


# --------------------------------------------------------------------------
# Representation B: bit string <-> real vector
# --------------------------------------------------------------------------


def decode_bits(bits) -> tuple:
    """Decode 90 bits into three reals using the formula on p.65.

    value(i) = Min + i / (2^N - 1) * (Max - Min)
    where i is the unsigned integer represented by the N-bit substring.
    """
    lo, hi = BOUNDS
    values = []
    for v in range(N_VARS):
        chunk = bits[v * N_BITS:(v + 1) * N_BITS]
        # most-significant-bit first
        i = 0
        for b in chunk:
            i = (i << 1) | int(b)
        values.append(lo + i / (2 ** N_BITS - 1) * (hi - lo))
    return tuple(values)


def encode_values(values) -> list:
    """Inverse of :func:`decode_bits`, for initialisation and reporting."""
    lo, hi = BOUNDS
    bits = []
    for value in values:
        value = min(max(value, lo), hi)
        i = int(round((value - lo) / (hi - lo) * (2 ** N_BITS - 1)))
        bits.extend(int(b) for b in format(i, f"0{N_BITS}b"))
    return bits


# --------------------------------------------------------------------------
# GA drivers
# --------------------------------------------------------------------------


def run_float_ga(pop_size: int, ngen: int, cxpb: float, mutpb: float,
                 seed: int, sigma: float = 0.2, mu: float = 0.0):
    from deap import algorithms, base, creator, tools

    if not hasattr(creator, "FitnessMinFloat"):
        creator.create("FitnessMinFloat", base.Fitness, weights=(-1.0,))
        creator.create("IndividualFloat", list, fitness=creator.FitnessMinFloat)

    random = __import__("random")
    random.seed(seed)

    toolbox = base.Toolbox()
    lo, hi = BOUNDS
    toolbox.register("attr_float", random.uniform, lo, hi)
    toolbox.register("individual", tools.initRepeat, creator.IndividualFloat,
                     toolbox.attr_float, n=N_VARS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (objective(*ind),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    # Manual loop so that the number of *evaluations* can be counted (the deck
    # asks for the number of fitness function invocations in Exercise 2).
    import copy

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    nevals_total = len(pop)
    logbook.record(gen=0, nevals=len(pop), **stats.compile(pop))

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        nevals_total += len(invalid)

        pop[:] = offspring
        hof.update(pop)
        logbook.record(gen=gen, nevals=len(invalid), **stats.compile(pop))

    return pop, logbook, hof, nevals_total


def run_bit_ga(pop_size: int, ngen: int, cxpb: float, mutpb: float,
               seed: int, indpb: float = 0.01):
    from deap import algorithms, base, creator, tools

    if not hasattr(creator, "FitnessMinBit"):
        creator.create("FitnessMinBit", base.Fitness, weights=(-1.0,))
        creator.create("IndividualBit", list, fitness=creator.FitnessMinBit)

    random = __import__("random")
    random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualBit,
                     toolbox.attr_bit, n=IND_LEN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # The decode step required by p.65 happens inside the fitness function.
    toolbox.register("evaluate", lambda ind: (objective(*decode_bits(ind)),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    import copy

    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit
    hof.update(pop)
    nevals_total = len(pop)
    logbook.record(gen=0, nevals=len(pop), **stats.compile(pop))

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        nevals_total += len(invalid)

        pop[:] = offspring
        hof.update(pop)
        logbook.record(gen=gen, nevals=len(invalid), **stats.compile(pop))

    return pop, logbook, hof, nevals_total


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    os.makedirs(LAB5_RESULTS, exist_ok=True)
    os.makedirs(LAB5_FIGURES, exist_ok=True)

    out = {"lab": 5, "exercise": 1, "title": "Function optimisation with a GA"}

    print("=" * 74)
    print("LAB 5, Exercise 1: minimise f(x,y,z) = (1.5+sin z)(sqrt((20-x)^2+(30-y)^2)+1)")
    print(f"domain [-250, 250]^3")
    print("=" * 74)
    print(f"analytic minimum: f{ANALYTIC_ARGMIN} = {ANALYTIC_MIN:.6f}")
    print(f"  (the distance term vanishes at (20,30) and sin(z) = -1 at z = -pi/2)\n")

    # sanity-check the decoder round-trips
    probe = (20.0, 30.0, -np.pi / 2)
    rt = decode_bits(encode_values(probe))
    print(f"decoder round-trip on {probe}: {tuple(round(v,6) for v in rt)}")
    print(f"  quantisation step = {(BOUNDS[1]-BOUNDS[0])/(2**N_BITS-1):.3e}\n")

    POP, NGEN, CXPB, MUTPB = 300, 100, 0.5, 0.2
    SEEDS = [1, 2, 3, 4, 5]

    results = {"float": [], "bit": []}
    logs = {"float": [], "bit": []}

    for label, runner in (("float", run_float_ga), ("bit", run_bit_ga)):
        print(f"--- representation: {label} "
              f"({POP} individuals, {NGEN} generations, Pc={CXPB}, Pm={MUTPB}) ---")
        for seed in SEEDS:
            t0 = time.time()
            pop, logbook, hof, nevals = runner(POP, NGEN, CXPB, MUTPB, seed)
            dt = time.time() - t0
            best = hof[0]
            if label == "float":
                best_values = tuple(float(v) for v in best)
            else:
                best_values = tuple(float(v) for v in decode_bits(best))
            best_f = float(best.fitness.values[0])
            results[label].append({
                "seed": seed,
                "best_fitness": best_f,
                "x": best_values[0], "y": best_values[1], "z": best_values[2],
                "fitness_invocations": int(nevals),
                "wall_clock_s": round(dt, 2),
                "gap_to_analytic": best_f - ANALYTIC_MIN,
            })
            logs[label].append([float(v) for v in logbook.select("min")])
            print(f"  seed {seed}: best f = {best_f:.6f}  at "
                  f"({best_values[0]:8.3f}, {best_values[1]:8.3f}, {best_values[2]:8.3f})  "
                  f"evals = {nevals:6d}  gap = {best_f - ANALYTIC_MIN:.2e}  [{dt:.1f}s]")

        arr = [r["best_fitness"] for r in results[label]]
        print(f"  --> mean best fitness {np.mean(arr):.6f} +/- {np.std(arr):.6f}, "
              f"mean evals {np.mean([r['fitness_invocations'] for r in results[label]]):.0f}")

    # ------------------------------------------------- mutation sensitivity
    # The exercise prescribes sigma=0.2 for the float representation and the DEAP
    # default indpb=0.01 for the bit representation.  That bit default turns out
    # to be pathologically low for this problem, which is a finding worth
    # documenting rather than hiding: with 90 bits and indpb=0.01 only ~0.9 bits
    # flip per mutation, so the population stalls in local optima.
    print("\n--- sensitivity to the mutation parameters ---")
    sens = []
    for indpb in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        b = []
        for seed in SEEDS:
            _, _, hof, _ = run_bit_ga(POP, NGEN, CXPB, MUTPB, seed, indpb=indpb)
            b.append(float(hof[0].fitness.values[0]))
        b = np.array(b)
        hit = int((b <= ANALYTIC_MIN * 1.01).sum())
        sens.append({"representation": "bit", "parameter": "indpb", "value": indpb,
                     "mean_best": float(b.mean()), "std_best": float(b.std()),
                     "min_best": float(b.min()), "runs_within_1pct": hit})
        print(f"  bit   indpb={indpb:<5} mean best={b.mean():.6f} sd={b.std():.6f} "
              f"min={b.min():.6f}  within 1% of optimum: {hit}/{len(SEEDS)}")
    for sigma in (0.05, 0.2, 1.0, 5.0):
        b = []
        for seed in SEEDS:
            _, _, hof, _ = run_float_ga(POP, NGEN, CXPB, MUTPB, seed, sigma=sigma)
            b.append(float(hof[0].fitness.values[0]))
        b = np.array(b)
        hit = int((b <= ANALYTIC_MIN * 1.01).sum())
        sens.append({"representation": "float", "parameter": "sigma", "value": sigma,
                     "mean_best": float(b.mean()), "std_best": float(b.std()),
                     "min_best": float(b.min()), "runs_within_1pct": hit})
        print(f"  float sigma={sigma:<5} mean best={b.mean():.6f} sd={b.std():.6f} "
              f"min={b.min():.6f}  within 1% of optimum: {hit}/{len(SEEDS)}")
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex1_mutation_sensitivity.csv"), sens,
              ["representation", "parameter", "value", "mean_best", "std_best",
               "min_best", "runs_within_1pct"])
    out["mutation_sensitivity"] = sens

    # ------------------------------------------------------- convergence stats
    print("\n--- convergence speed ---")
    conv_rows = []
    for label in ("float", "bit"):
        curves = np.array(logs[label])           # (seeds, generations+1)
        for gen in range(curves.shape[1]):
            conv_rows.append({
                "representation": label,
                "generation": gen,
                "best_min_mean": float(curves[:, gen].mean()),
                "best_min_std": float(curves[:, gen].std()),
                "best_min_min": float(curves[:, gen].min()),
            })
        # First generation reaching within a tolerance of the analytic minimum.
        # Two tolerances are reported because "within 1%" of an optimum of 0.5 is a
        # very tight absolute band (0.005), so a relative tolerance alone is harsh.
        for tol_label, tol in (("within_1pct", 0.01), ("within_0.1pct", 0.001)):
            target = ANALYTIC_MIN + tol * abs(ANALYTIC_MIN)
            reached = [int(np.argmax(c <= target)) if np.any(c <= target) else -1
                       for c in curves]
            hit = [r for r in reached if r >= 0]
            print(f"  {label:5s} {tol_label:12s}: first generation per seed = {reached}"
                  f"  ({len(hit)}/{len(reached)} runs reached it; "
                  f"mean generation = {np.mean(hit):.1f})" if hit else
                  f"  {label:5s} {tol_label:12s}: never reached in any run")

    write_csv(os.path.join(LAB5_RESULTS, "ga_ex1_convergence.csv"), conv_rows,
              ["representation", "generation", "best_min_mean", "best_min_std",
               "best_min_min"])

    run_rows = []
    for label in ("float", "bit"):
        for r in results[label]:
            run_rows.append({"representation": label, **r})
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex1_runs.csv"), run_rows,
              ["representation", "seed", "best_fitness", "x", "y", "z",
               "fitness_invocations", "wall_clock_s", "gap_to_analytic"])

    out["settings"] = {"pop_size": POP, "generations": NGEN, "cxpb": CXPB,
                       "mutpb": MUTPB, "seeds": SEEDS, "n_bits_per_var": N_BITS,
                       "bounds": list(BOUNDS),
                       "analytic_min": ANALYTIC_MIN,
                       "analytic_argmin": list(ANALYTIC_ARGMIN)}
    out["results"] = results
    out["convergence_summary"] = conv_rows
    for label in ("float", "bit"):
        arr = [r["best_fitness"] for r in results[label]]
        ev = [r["fitness_invocations"] for r in results[label]]
        out[f"{label}_mean_best"] = float(np.mean(arr))
        out[f"{label}_std_best"] = float(np.std(arr))
        out[f"{label}_mean_evals"] = float(np.mean(ev))

    with open(os.path.join(LAB5_RESULTS, "lab5_ex1_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colours = {"float": "C0", "bit": "C1"}
    for label in ("float", "bit"):
        curves = np.array(logs[label])
        gens = np.arange(curves.shape[1])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        axes[0].plot(gens, mean, "-", color=colours[label],
                     label=f"{label} representation (mean of {len(curves)} runs)")
        axes[0].fill_between(gens, mean - std, mean + std, color=colours[label], alpha=0.18)
    axes[0].axhline(ANALYTIC_MIN, color="k", ls="--", lw=1.2,
                    label=f"analytic minimum = {ANALYTIC_MIN:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("best fitness (log scale)")
    axes[0].set_title("Convergence: best fitness vs generation")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    labels = ["float", "bit"]
    means = [np.mean([r["fitness_invocations"] for r in results[l]]) for l in labels]
    stds = [np.std([r["fitness_invocations"] for r in results[l]]) for l in labels]
    axes[1].bar(labels, means, yerr=stds, capsize=5,
                color=[colours[l] for l in labels], alpha=0.8)
    axes[1].set_ylabel("fitness evaluations to reach generation 100")
    axes[1].set_title("Computational effort per representation")
    axes[1].grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Lab 5 Ex.1 - float vs bit representation for GA function minimisation",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB5_FIGURES, "ga_ex1_convergence.png"), dpi=150)
    plt.close(fig)
    print("\nwrote lab5/figures/ga_ex1_convergence.png")
    print("wrote lab5/results/ga_ex1_runs.csv, lab5/results/ga_ex1_convergence.csv")
    print("wrote lab5/results/lab5_ex1_summary.json")


if __name__ == "__main__":
    main()
