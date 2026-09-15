# ============================================================
# Particle Swarm Optimization - DEAP
#
# Exercise:
#
# f(x,y,z) =
# (1 - cos(2*pi / (1 + exp(-(x-125)^2 - (y-1.27)^2))))
# ------------------------------------------------------------
# / (1 + z^2)
#
# 1. 50 particles:
#       search good values for w, c1, c2
#
# 2. 100 particles:
#       use the best w, c1, c2 found in step 1
#
# 3. 100 particles:
#       search again for better parameter values
#
# Maximum number of iterations = 10000
# ============================================================

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from deap import base
from deap import creator
from deap import tools


# ============================================================
# ALGORITHM PARAMETERS
# ============================================================

RSEED = 42

# Maximum number of iterations
GEN = 10000

# Search domain
PMIN = -250
PMAX = 250

# Particle speed limits
SMIN = -50
SMAX = 50

# Number of variables: x, y, z
PSIZE = 3


# ============================================================
# DEAP CLASSES
# ============================================================

# We want to MAXIMIZE the function
#
# weights=(1.0,) means maximization
#
if not hasattr(creator, "FitnessMaxPSO"):
    creator.create(
        "FitnessMaxPSO",
        base.Fitness,
        weights=(1.0,)
    )


if not hasattr(creator, "ParticlePSO"):
    creator.create(
        "ParticlePSO",
        list,
        fitness=creator.FitnessMaxPSO,
        speed=None,
        smin=None,
        smax=None,
        pmin=None,
        pmax=None,
        best=None
    )


# ============================================================
# FITNESS FUNCTION
# ============================================================

def evalfun(individual):

    x, y, z = individual

    # --------------------------------------------------------
    # f(x,y,z) =
    #
    # (1 - cos(
    #       2*pi /
    #       (1 + exp(-(x-125)^2 - (y-1.27)^2))
    # ))
    #
    # /
    #
    # (1 + z^2)
    # --------------------------------------------------------

    exponent = (
        -(x - 125) ** 2
        -(y - 1.27) ** 2
    )

    # Avoid numerical overflow in exp()
    if exponent < -700:
        exp_value = 0.0
    else:
        exp_value = math.exp(exponent)

    denominator = 1.0 + exp_value

    value = (
        1.0
        - math.cos(
            2.0 * math.pi / denominator
        )
    ) / (
        1.0 + z ** 2
    )

    return (value,)


# ============================================================
# PARTICLE GENERATION
# ============================================================

def generate_particle(
    size,
    pmin,
    pmax,
    smin,
    smax
):

    particle = creator.ParticlePSO(
        random.uniform(pmin, pmax)
        for _ in range(size)
    )

    particle.speed = [
        random.uniform(smin, smax)
        for _ in range(size)
    ]

    particle.smin = smin
    particle.smax = smax

    particle.pmin = pmin
    particle.pmax = pmax

    return particle


# ============================================================
# PARTICLE UPDATE
# ============================================================

def update_particle(
    particle,
    best,
    w,
    c1,
    c2
):

    for i in range(len(particle)):

        # Random coefficients
        r1 = random.random()
        r2 = random.random()

        # ----------------------------------------------------
        # Cognitive component
        #
        # c1 * r1 * (personal_best - current_position)
        # ----------------------------------------------------

        cognitive = (
            c1
            * r1
            * (
                particle.best[i]
                - particle[i]
            )
        )

        # ----------------------------------------------------
        # Social component
        #
        # c2 * r2 * (global_best - current_position)
        # ----------------------------------------------------

        social = (
            c2
            * r2
            * (
                best[i]
                - particle[i]
            )
        )

        # ----------------------------------------------------
        # Velocity update
        # ----------------------------------------------------

        particle.speed[i] = (
            w * particle.speed[i]
            + cognitive
            + social
        )

        # ----------------------------------------------------
        # Speed limits
        # ----------------------------------------------------

        if particle.speed[i] < particle.smin:

            particle.speed[i] = particle.smin

        elif particle.speed[i] > particle.smax:

            particle.speed[i] = particle.smax

        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        particle[i] += particle.speed[i]

        # ----------------------------------------------------
        # Position limits
        # ----------------------------------------------------

        if particle[i] < particle.pmin:

            particle[i] = particle.pmin

        elif particle[i] > particle.pmax:

            particle[i] = particle.pmax


# ============================================================
# RUN ONE PSO EXPERIMENT
# ============================================================

def run_pso(
    population_size,
    w,
    c1,
    c2,
    generations=GEN,
    seed=RSEED,
    verbose=False
):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------------
    # Toolbox
    # --------------------------------------------------------

    toolbox = base.Toolbox()

    toolbox.register(
        "particle",
        generate_particle,
        size=PSIZE,
        pmin=PMIN,
        pmax=PMAX,
        smin=SMIN,
        smax=SMAX
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.particle
    )

    toolbox.register(
        "evaluate",
        evalfun
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = toolbox.population(
        n=population_size
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = tools.Statistics(
        key=lambda individual:
        individual.fitness.values
    )

    stats.register(
        "avg",
        np.mean
    )

    stats.register(
        "std",
        np.std
    )

    stats.register(
        "min",
        np.min
    )

    stats.register(
        "max",
        np.max
    )

    # --------------------------------------------------------
    # Logbook
    # --------------------------------------------------------

    logbook = tools.Logbook()

    logbook.header = [
        "gen",
        "evals",
        "avg",
        "std",
        "min",
        "max"
    ]

    # Global best
    global_best = None

    # --------------------------------------------------------
    # MAIN PSO LOOP
    # --------------------------------------------------------

    for generation in range(generations):

        # ====================================================
        # Evaluate particles
        # ====================================================

        for particle in population:

            particle.fitness.values = (
                toolbox.evaluate(particle)
            )

            # ------------------------------------------------
            # Personal best
            # ------------------------------------------------

            if (
                particle.best is None
                or particle.best.fitness
                < particle.fitness
            ):

                particle.best = (
                    creator.ParticlePSO(
                        particle
                    )
                )

                particle.best.fitness.values = (
                    particle.fitness.values
                )

            # ------------------------------------------------
            # Global best
            # ------------------------------------------------

            if (
                global_best is None
                or global_best.fitness
                < particle.fitness
            ):

                global_best = (
                    creator.ParticlePSO(
                        particle
                    )
                )

                global_best.fitness.values = (
                    particle.fitness.values
                )

        # ====================================================
        # Statistics
        # ====================================================

        record = stats.compile(
            population
        )

        logbook.record(
            gen=generation,
            evals=len(population),
            **record
        )

        # ====================================================
        # Optional progress output
        # ====================================================

        if verbose:

            if (
                generation % 500 == 0
                or generation == generations - 1
            ):

                print(
                    f"Generation {generation:5d} | "
                    f"Best = "
                    f"{global_best.fitness.values[0]:.12f}"
                )

        # ====================================================
        # Update particles
        # ====================================================

        for particle in population:

            update_particle(
                particle,
                global_best,
                w,
                c1,
                c2
            )

    return (
        global_best,
        logbook
    )


# ============================================================
# SEARCH PARAMETERS
# ============================================================

def parameter_search(
    population_size,
    w_values,
    c1_values,
    c2_values,
    generations=GEN,
    seed=RSEED
):

    results = []

    total_experiments = (
        len(w_values)
        * len(c1_values)
        * len(c2_values)
    )

    current_experiment = 0

    print()
    print("=" * 75)
    print(
        f"PARAMETER SEARCH - "
        f"{population_size} PARTICLES"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Try every combination
    # --------------------------------------------------------

    for w in w_values:

        for c1 in c1_values:

            for c2 in c2_values:

                current_experiment += 1

                print(
                    f"\nExperiment "
                    f"{current_experiment}/"
                    f"{total_experiments}"
                )

                print(
                    f"w={w:.2f}, "
                    f"c1={c1:.2f}, "
                    f"c2={c2:.2f}"
                )

                start_time = time.time()

                best, logbook = run_pso(
                    population_size=
                    population_size,

                    w=w,
                    c1=c1,
                    c2=c2,

                    generations=
                    generations,

                    seed=seed,

                    verbose=False
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                fitness = (
                    best.fitness.values[0]
                )

                result = {

                    "population":
                    population_size,

                    "w": w,

                    "c1": c1,

                    "c2": c2,

                    "best_fitness":
                    fitness,

                    "x": best[0],

                    "y": best[1],

                    "z": best[2],

                    "time":
                    elapsed
                }

                results.append(result)

                print(
                    f"Best fitness = "
                    f"{fitness:.12f}"
                )

                print(
                    f"Time = "
                    f"{elapsed:.2f} s"
                )

    # --------------------------------------------------------
    # Sort by fitness
    #
    # Because we maximize, the highest value comes first.
    # --------------------------------------------------------

    results.sort(
        key=lambda result:
        result["best_fitness"],
        reverse=True
    )

    return results


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

def save_results(
    filename,
    results
):

    if len(results) == 0:
        return

    fieldnames = [
        "population",
        "w",
        "c1",
        "c2",
        "best_fitness",
        "x",
        "y",
        "z",
        "time"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\nResults saved in: "
        f"{filename}"
    )


# ============================================================
# PRINT BEST RESULTS
# ============================================================

def print_best_results(
    results,
    title,
    number=5
):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    for i, result in enumerate(
        results[:number],
        start=1
    ):

        print(
            f"{i}. "
            f"fitness="
            f"{result['best_fitness']:.12f} | "
            f"w={result['w']:.2f} | "
            f"c1={result['c1']:.2f} | "
            f"c2={result['c2']:.2f} | "
            f"x={result['x']:.6f} | "
            f"y={result['y']:.6f} | "
            f"z={result['z']:.6f}"
        )


# ============================================================
# PLOT CONVERGENCE
# ============================================================

def plot_convergence(
    logbook,
    title,
    filename
):

    generations = logbook.select(
        "gen"
    )

    best = logbook.select(
        "max"
    )

    average = logbook.select(
        "avg"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        generations,
        best,
        label="Best fitness"
    )

    plt.plot(
        generations,
        average,
        label="Average fitness"
    )

    plt.xlabel(
        "Generation"
    )

    plt.ylabel(
        "Fitness"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.show()


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_comparison(
    log_50,
    log_100,
    title,
    filename
):

    generations_50 = (
        log_50.select("gen")
    )

    best_50 = (
        log_50.select("max")
    )

    generations_100 = (
        log_100.select("gen")
    )

    best_100 = (
        log_100.select("max")
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        generations_50,
        best_50,
        label="50 particles"
    )

    plt.plot(
        generations_100,
        best_100,
        label="100 particles"
    )

    plt.xlabel(
        "Generation"
    )

    plt.ylabel(
        "Best fitness"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("PARTICLE SWARM OPTIMIZATION")
    print("=" * 75)

    print(
        f"\nMaximum iterations: {GEN}"
    )

    print(
        f"Search domain: "
        f"[{PMIN}, {PMAX}]"
    )

    # ========================================================
    # STEP 1
    #
    # 50 PARTICLES
    #
    # Search good values of w, c1, c2
    # ========================================================

    print()
    print("=" * 75)
    print("STEP 1 - 50 PARTICLES")
    print("=" * 75)

    # --------------------------------------------------------
    # Values inside the required ranges:
    #
    # w  in [0,1]
    # c1 in [0,2]
    # c2 in [0,2]
    #
    # You can make this grid finer if necessary.
    # --------------------------------------------------------

    w_values = [
        0.2,
        0.4,
        0.6,
        0.7,
        0.8,
        0.9
    ]

    c1_values = [
        0.5,
        1.0,
        1.5,
        2.0
    ]

    c2_values = [
        0.5,
        1.0,
        1.5,
        2.0
    ]

    results_50 = parameter_search(
        population_size=50,

        w_values=w_values,

        c1_values=c1_values,

        c2_values=c2_values,

        generations=GEN,

        seed=RSEED
    )

    # --------------------------------------------------------
    # Print best configurations
    # --------------------------------------------------------

    print_best_results(
        results_50,
        "STEP 1 - BEST RESULTS WITH 50 PARTICLES"
    )

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    save_results(
        "pso_results_50.csv",
        results_50
    )

    # --------------------------------------------------------
    # Best parameters found
    # --------------------------------------------------------

    best_50 = results_50[0]

    best_w = best_50["w"]
    best_c1 = best_50["c1"]
    best_c2 = best_50["c2"]

    print()
    print(
        "Best parameters found with "
        "50 particles:"
    )

    print(
        f"w  = {best_w}"
    )

    print(
        f"c1 = {best_c1}"
    )

    print(
        f"c2 = {best_c2}"
    )

    print(
        f"Fitness = "
        f"{best_50['best_fitness']:.12f}"
    )

    # ========================================================
    # STEP 2
    #
    # 100 PARTICLES
    #
    # Use parameters found in step 1
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STEP 2 - 100 PARTICLES "
        "WITH PARAMETERS FROM STEP 1"
    )
    print("=" * 75)

    best_100_step2, log_100_step2 = run_pso(

        population_size=100,

        w=best_w,

        c1=best_c1,

        c2=best_c2,

        generations=GEN,

        seed=RSEED,

        verbose=True
    )

    fitness_100_step2 = (
        best_100_step2
        .fitness
        .values[0]
    )

    print()
    print(
        "Best result with 100 particles:"
    )

    print(
        f"x = "
        f"{best_100_step2[0]:.10f}"
    )

    print(
        f"y = "
        f"{best_100_step2[1]:.10f}"
    )

    print(
        f"z = "
        f"{best_100_step2[2]:.10f}"
    )

    print(
        f"f(x,y,z) = "
        f"{fitness_100_step2:.12f}"
    )

    # --------------------------------------------------------
    # Plot convergence
    # --------------------------------------------------------

    plot_convergence(

        log_100_step2,

        (
            "PSO convergence - 100 particles\n"
            f"w={best_w}, "
            f"c1={best_c1}, "
            f"c2={best_c2}"
        ),

        "pso_convergence_100_step2.png"
    )

    # ========================================================
    # STEP 3
    #
    # 100 PARTICLES
    #
    # Search again for potentially better parameters
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STEP 3 - SEARCH AGAIN "
        "WITH 100 PARTICLES"
    )
    print("=" * 75)

    results_100 = parameter_search(

        population_size=100,

        w_values=w_values,

        c1_values=c1_values,

        c2_values=c2_values,

        generations=GEN,

        seed=RSEED
    )

    # --------------------------------------------------------
    # Print best configurations
    # --------------------------------------------------------

    print_best_results(
        results_100,
        "STEP 3 - BEST RESULTS WITH 100 PARTICLES"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        "pso_results_100.csv",
        results_100
    )

    # --------------------------------------------------------
    # Best 100-particle configuration
    # --------------------------------------------------------

    best_100 = results_100[0]

    print()
    print(
        "Best parameters found with "
        "100 particles:"
    )

    print(
        f"w  = {best_100['w']}"
    )

    print(
        f"c1 = {best_100['c1']}"
    )

    print(
        f"c2 = {best_100['c2']}"
    )

    print(
        f"Fitness = "
        f"{best_100['best_fitness']:.12f}"
    )

    # ========================================================
    # FINAL RUN WITH BEST 100-PARTICLE PARAMETERS
    # ========================================================

    print()
    print("=" * 75)
    print(
        "FINAL RUN - BEST 100-PARTICLE PARAMETERS"
    )
    print("=" * 75)

    final_best, final_log = run_pso(

        population_size=100,

        w=best_100["w"],

        c1=best_100["c1"],

        c2=best_100["c2"],

        generations=GEN,

        seed=RSEED,

        verbose=True
    )

    final_fitness = (
        final_best
        .fitness
        .values[0]
    )

    # --------------------------------------------------------
    # Final solution
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL SOLUTION")
    print("=" * 75)

    print(
        f"x = {final_best[0]:.12f}"
    )

    print(
        f"y = {final_best[1]:.12f}"
    )

    print(
        f"z = {final_best[2]:.12f}"
    )

    print(
        f"f(x,y,z) = "
        f"{final_fitness:.12f}"
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"50 particles:"
    )

    print(
        f"    w={best_50['w']}, "
        f"c1={best_50['c1']}, "
        f"c2={best_50['c2']}"
    )

    print(
        f"    fitness="
        f"{best_50['best_fitness']:.12f}"
    )

    print()

    print(
        f"100 particles "
        f"(parameters from step 1):"
    )

    print(
        f"    w={best_w}, "
        f"c1={best_c1}, "
        f"c2={best_c2}"
    )

    print(
        f"    fitness="
        f"{fitness_100_step2:.12f}"
    )

    print()

    print(
        f"100 particles "
        f"(re-optimized parameters):"
    )

    print(
        f"    w={best_100['w']}, "
        f"c1={best_100['c1']}, "
        f"c2={best_100['c2']}"
    )

    print(
        f"    fitness="
        f"{best_100['best_fitness']:.12f}"
    )

    # ========================================================
    # CONVERGENCE GRAPH FOR BEST CONFIGURATION
    # ========================================================

    plot_convergence(

        final_log,

        (
            "Best PSO configuration - "
            "100 particles"
        ),

        "pso_convergence_best_100.png"
    )

    # ========================================================
    # COMPARISON 50 vs 100 PARTICLES
    #
    # Re-run the best 50-particle configuration so that
    # we have its complete logbook.
    # ========================================================

    best_particle_50, log_50 = run_pso(

        population_size=50,

        w=best_50["w"],

        c1=best_50["c1"],

        c2=best_50["c2"],

        generations=GEN,

        seed=RSEED,

        verbose=False
    )

    plot_comparison(

        log_50,

        final_log,

        "PSO convergence: 50 vs 100 particles",

        "pso_comparison_50_vs_100.png"
    )

    # ========================================================
    # THEORETICAL MAXIMUM
    # ========================================================
    #
    # The numerator is in [0,2] and the denominator is
    # >= 1.
    #
    # The theoretical maximum is therefore 2, if the
    # numerator can reach 2 and z=0.
    #
    # The PSO result can be compared with this value.
    # ========================================================

    theoretical_max = 2.0

    print()
    print("=" * 75)
    print("THEORETICAL MAXIMUM")
    print("=" * 75)

    print(
        f"Theoretical maximum = "
        f"{theoretical_max:.12f}"
    )

    print(
        f"PSO maximum found    = "
        f"{final_fitness:.12f}"
    )

    print(
        f"Absolute error       = "
        f"{abs(theoretical_max - final_fitness):.12f}"
    )

    print()
    print("Experiments completed.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()