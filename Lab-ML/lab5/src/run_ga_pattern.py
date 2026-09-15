"""Lab 5 - Exercise 2: pattern "guessing" with a GA (deck pp. 66-68).

The task is to recover a hidden binary pattern by searching for an individual
that matches it as closely as possible.  The deck notes this is *the same problem
as OneMax*: with fitness = number of matching bits, the global optimum is the
pattern itself.

Key points from the deck:

* The pattern has 16 x 14 = 224 bits, so random guessing would need ~2^223
  attempts on average (the deck's "2^223" and "6.8e8 years" figures).
* A **fitness-guided** algorithm needs at most ~225 attempts, because each
  generation of a 225-individual population can, in the ideal case, fix one more
  bit.  Reproducing that bound is the point of the exercise.
* Exercise 2.2 (p.68) asks how the number of fitness evaluations N_f needed to
  find the solution grows with pattern size, averaged over 5 runs per pattern,
  and then which (population size, generations) combination is best.

Outputs
-------
results/ga_ex2_smiley.json      the smiley run: N_f over 5 repetitions
results/ga_ex2_scaling.csv      N_f vs pattern size
results/ga_ex2_parameters.csv   N_f vs (pop_size, generations)
figures/ga_ex2_convergence.png  best/average fitness vs generation for the smiley
figures/ga_ex2_scaling.png      N_f vs pattern size, with the theoretical bound
"""

from __future__ import annotations

import json
import os
import sys

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

SMILEY = os.path.join(LAB1_DIR, "..", "ml-patterns", "deap", "smiley.txt")
N_REPEATS = 5


def read_pattern(path: str) -> np.ndarray:
    """Read a pattern file: first two lines are the dimensions, then the grid."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    try:
        height, width = int(lines[0]), int(lines[1])
        grid = np.array([[int(v) for v in ln.split()] for ln in lines[2:]], dtype=int)
        if grid.shape != (height, width):
            # tolerate a file whose header disagrees with the body
            height, width = grid.shape
    except ValueError:
        # no header: whole file is the grid
        grid = np.array([[int(v) for v in ln.split()] for ln in lines], dtype=int)
        height, width = grid.shape
    return grid.reshape(-1)


def make_random_pattern(n_bits: int, seed: int, density: float = 0.25) -> np.ndarray:
    """A random target with a plausible ON density (the smiley is ~25 % ones).

    A pattern that is nearly all-zero is trivially easy, so the density is
    matched to the smiley for a meaningful comparison.
    """
    rng = np.random.default_rng(seed)
    return (rng.random(n_bits) < density).astype(int)


#: Per-bit mutation probability.  This matters enormously: DEAP's default of
#: 1/n_bits and the value used in Exercise 1 (0.05) both fail on a 224-bit
#: pattern.  0.005 solves all runs with the fewest evaluations (see the
#: sensitivity table below), so it is the default here.
BEST_INDPB = 0.005


def run_ga(target: np.ndarray, pop_size: int, ngen: int, seed: int,
           cxpb: float = 0.5, mutpb: float = 0.2, indpb: float = BEST_INDPB,
           count_evaluations: bool = True):
    """Maximise the number of bits matching ``target``.

    Returns (best_individual, best_fitness, n_evaluations, logbook_min_history).
    The GA stops early once the perfect solution is found, so ``n_evaluations``
    is the number of fitness invocations actually needed (N_f in the deck).
    """
    from deap import base, creator, tools

    n_bits = len(target)

    if not hasattr(creator, "FitnessMaxPat"):
        creator.create("FitnessMaxPat", base.Fitness, weights=(1.0,))
        creator.create("IndividualPat", list, fitness=creator.FitnessMaxPat)

    import random as _random
    _random.seed(seed)

    def evaluate(ind):
        # fitness = number of matching bits (OneMax on the XOR with the target)
        return (int(np.sum(np.asarray(ind) == target)),)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit", _random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualPat,
                     toolbox.attr_bit, n=n_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    # selection is applied inline below so that the evaluation count N_f can be
    # tracked exactly; no toolbox registration is needed for it.

    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit
    n_evals = len(pop)

    best = max(pop, key=lambda i: i.fitness.values[0])
    history = [float(best.fitness.values[0])]

    import copy

    for gen in range(1, ngen + 1):
        if best.fitness.values[0] == n_bits:
            break
        offspring = [copy.deepcopy(i) for i in tools.selTournament(pop, len(pop), 3)]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if _random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if _random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        n_evals += len(invalid)
        pop[:] = offspring
        gen_best = max(pop, key=lambda i: i.fitness.values[0])
        if gen_best.fitness.values[0] > best.fitness.values[0]:
            best = gen_best
        history.append(float(best.fitness.values[0]))

    return best, int(best.fitness.values[0]), n_evals, history


def main() -> None:
    os.makedirs(LAB5_RESULTS, exist_ok=True)
    os.makedirs(LAB5_FIGURES, exist_ok=True)

    out = {"lab": 5, "exercise": 2, "title": "Pattern guessing with a GA"}

    target = read_pattern(SMILEY)
    n_bits = len(target)
    print("=" * 74)
    print("LAB 5, Exercise 2: recover the pattern in smiley.txt")
    print("=" * 74)
    print(f"pattern: 16 x 14 = {n_bits} bits, {int(target.sum())} ON "
          f"({100*target.mean():.1f} %)")
    print(f"brute force would need ~2^{n_bits-1} attempts on average "
          f"(the deck's figure)")
    print(f"a fitness-guided search should need at most ~{n_bits + 1} attempts\n")

    # ---------------------------------------------------- 2.1 the smiley run
    # The deck's bound of ~225 comes from a 225-individual population, one
    # generation.
    POP, NGEN = n_bits + 1, 200  # 225 individuals, up to 200 generations
    print(f"--- smiley.txt, population = {POP}, max generations = {NGEN}, "
          f"{N_REPEATS} repetitions ---")
    runs, histories = [], []
    for rep in range(1, N_REPEATS + 1):
        best, fit, nevals, hist = run_ga(target, POP, NGEN, seed=rep)
        solved = fit == n_bits
        runs.append({"repetition": rep, "best_fitness": fit, "n_bits": n_bits,
                     "solved": solved, "fitness_invocations": nevals,
                     "proportion_matched": fit / n_bits})
        histories.append(hist)
        print(f"  run {rep}: matched {fit}/{n_bits} bits "
              f"({'SOLVED' if solved else 'NOT SOLVED'}), "
              f"N_f = {nevals} evaluations")

    solved = [r for r in runs if r["solved"]]
    nf = [r["fitness_invocations"] for r in solved] or [r["fitness_invocations"] for r in runs]
    print(f"\n  solved {len(solved)}/{N_REPEATS} runs")
    print(f"  mean N_f over solved runs = {np.mean(nf):.1f} "
          f"(deck's theoretical bound ~{n_bits + 1})")
    print(f"  mean N_f over all runs    = "
          f"{np.mean([r['fitness_invocations'] for r in runs]):.1f}")

    out["smiley"] = {
        "n_bits": n_bits,
        "n_ones": int(target.sum()),
        "on_density": float(target.mean()),
        "population": POP,
        "max_generations": NGEN,
        "repetitions": N_REPEATS,
        "runs": runs,
        "n_solved": len(solved),
        "mean_Nf_solved": float(np.mean(nf)),
        "deck_bound": n_bits + 1,
    }

    # ------------------------------------------------- 2.2 scaling with size
    print("\n--- 2.2 growth of N_f with pattern size (5 runs each) ---")
    scaling = []
    sizes = [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24)]
    for h, w in sizes:
        n = h * w
        nf_list, solved_count = [], 0
        for rep in range(1, N_REPEATS + 1):
            tgt = make_random_pattern(n, seed=1000 + rep)
            # generations scaled to the problem so each run stays bounded
            best, fit, nevals, _ = run_ga(tgt, n + 1, min(2 * n, 400), seed=rep)
            nf_list.append(nevals)
            solved_count += int(fit == n)
        scaling.append({
            "pattern": f"{h}x{w}", "n_bits": n,
            "pop_size": n + 1,
            "mean_Nf": float(np.mean(nf_list)), "std_Nf": float(np.std(nf_list)),
            "min_Nf": int(np.min(nf_list)),
            "runs_solved": solved_count, "repetitions": N_REPEATS,
            "ratio_Nf_over_n": float(np.mean(nf_list) / n),
        })
        print(f"  {h:2d}x{w:<2d} ({n:4d} bits): mean N_f = {np.mean(nf_list):9.1f} "
              f"+/- {np.std(nf_list):8.1f}  ratio N_f/n = "
              f"{np.mean(nf_list)/n:6.2f}  solved {solved_count}/{N_REPEATS}")
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex2_scaling.csv"), scaling,
              ["pattern", "n_bits", "pop_size", "mean_Nf", "std_Nf", "min_Nf",
               "runs_solved", "repetitions", "ratio_Nf_over_n"])
    out["scaling"] = scaling

    # ------------------------------------- 2.2b sensitivity to the mutation rate
    print("\n--- 2.2b sensitivity to the per-bit mutation probability ---")
    sens = []
    for indpb in (1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2):
        fits, nev = [], []
        for rep in range(1, N_REPEATS + 1):
            _, fit, nevals, _ = run_ga(target, POP, 400, seed=rep, indpb=indpb)
            fits.append(fit)
            nev.append(nevals)
        f = np.array(fits)
        sens.append({"indpb": indpb, "n_solved": int((f == n_bits).sum()),
                     "mean_best_fitness": float(f.mean()),
                     "mean_Nf": float(np.mean(nev)),
                     "ratio_Nf_over_n": float(np.mean(nev) / n_bits)})
        print(f"  indpb={indpb:<8} solved {int((f==n_bits).sum())}/{N_REPEATS}  "
              f"mean best={f.mean():6.1f}/{n_bits}  mean N_f={np.mean(nev):9.0f}  "
              f"N_f/n={np.mean(nev)/n_bits:7.2f}")
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex2_mutation_sensitivity.csv"), sens,
              ["indpb", "n_solved", "mean_best_fitness", "mean_Nf", "ratio_Nf_over_n"])
    out["mutation_sensitivity"] = sens
    best_indpb = max(sens, key=lambda r: (r["n_solved"], -r["mean_Nf"]))
    print(f"  -> best indpb = {best_indpb['indpb']} "
          f"(solves {best_indpb['n_solved']}/{N_REPEATS}, "
          f"mean N_f = {best_indpb['mean_Nf']:.0f})")
    out["best_indpb"] = best_indpb

    # ------------------------------------------- 2.2c hill-climber reference
    # The deck asks in Exercise 2 whether a "very specific but smart" algorithm
    # could do better.  A (1+1) hill-climber flipping exactly ONE bit per
    # iteration is the natural candidate for OneMax, so it is measured here.
    print("\n--- 2.2c (1+1) hill-climber with single-bit flips, for comparison ---")
    hc = []
    for rep in range(1, N_REPEATS + 1):
        rng = np.random.default_rng(rep)
        cur = (rng.random(n_bits) < target.mean()).astype(int)
        curfit = int((cur == target).sum())
        evals = 1
        while curfit < n_bits and evals < 200000:
            i = int(rng.integers(n_bits))
            cand = cur.copy()
            cand[i] ^= 1
            evals += 1
            f2 = int((cand == target).sum())
            if f2 >= curfit:
                cur, curfit = cand, f2
        hc.append({"repetition": rep, "solved": curfit == n_bits,
                   "fitness_invocations": evals})
        print(f"  run {rep}: {'solved' if curfit == n_bits else 'failed'} "
              f"in N_f = {evals}")
    hc_nf = [r["fitness_invocations"] for r in hc if r["solved"]]
    print(f"  mean N_f over solved runs = {np.mean(hc_nf):.0f} "
          f"(vs {out['smiley']['mean_Nf_solved']:.0f} for the GA)")
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex2_hillclimber.csv"), hc,
              ["repetition", "solved", "fitness_invocations"])
    out["hillclimber"] = {"runs": hc, "mean_Nf_solved": float(np.mean(hc_nf))}

    # ------------------------------------------- 2.3 parameter combinations
    print("\n--- 2.3 effect of population size and generation count on smiley ---")
    params = []
    for pop in (50, 100, 225, 400):
        for ngen in (50, 100, 200):
            nf_list, solved_count = [], 0
            for rep in range(1, N_REPEATS + 1):
                best, fit, nevals, _ = run_ga(target, pop, ngen, seed=rep)
                nf_list.append(nevals)
                solved_count += int(fit == n_bits)
            params.append({"pop_size": pop, "max_generations": ngen,
                           "mean_Nf": float(np.mean(nf_list)),
                           "std_Nf": float(np.std(nf_list)),
                           "runs_solved": solved_count,
                           "repetitions": N_REPEATS})
            print(f"  pop={pop:4d} ngen={ngen:4d}: mean N_f = {np.mean(nf_list):9.1f} "
                  f" solved {solved_count}/{N_REPEATS}")
    write_csv(os.path.join(LAB5_RESULTS, "ga_ex2_parameters.csv"), params,
              ["pop_size", "max_generations", "mean_Nf", "std_Nf", "runs_solved",
               "repetitions"])
    out["parameters"] = params
    best_combo = max(params, key=lambda r: (r["runs_solved"], -r["mean_Nf"]))
    print(f"\n  best combination by (solved runs, then fewest evaluations): "
          f"pop={best_combo['pop_size']}, ngen={best_combo['max_generations']}, "
          f"mean N_f = {best_combo['mean_Nf']:.1f}, "
          f"solved {best_combo['runs_solved']}/{N_REPEATS}")
    out["best_combination"] = best_combo

    with open(os.path.join(LAB5_RESULTS, "ga_ex2_smiley.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # --------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for hist, rep in zip(histories, range(1, N_REPEATS + 1)):
        axes[0].plot(range(len(hist)), hist, alpha=0.8, label=f"run {rep}")
    axes[0].axhline(n_bits, color="k", ls="--", lw=1.2, label="perfect match")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("best fitness (bits matched)")
    axes[0].set_title("smiley.txt: best fitness vs generation")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    xs = [r["n_bits"] for r in scaling]
    ys = [r["mean_Nf"] for r in scaling]
    es = [r["std_Nf"] for r in scaling]
    axes[1].errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, label="measured mean $N_f$")
    axes[1].plot(xs, [x + 1 for x in xs], "k--", lw=1.2,
                 label="deck's bound ($N$ + 1)")
    axes[1].set_xlabel("pattern size (bits)")
    axes[1].set_ylabel("fitness evaluations $N_f$")
    axes[1].set_title("Growth of $N_f$ with pattern size")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle("Lab 5 Ex.2 - pattern guessing with a GA", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB5_FIGURES, "ga_ex2_pattern.png"), dpi=150)
    plt.close(fig)
    print("\nwrote lab5/figures/ga_ex2_pattern.png")
    print("wrote lab5/results/ga_ex2_smiley.json, ga_ex2_scaling.csv, "
          "ga_ex2_parameters.csv")


if __name__ == "__main__":
    main()
