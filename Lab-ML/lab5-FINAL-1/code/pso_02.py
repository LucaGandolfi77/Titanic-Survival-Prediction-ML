# ============================================================
# Particle Swarm Optimization (PSO) - DEAP
#
# Exercise
# ============================================================
#
# Maximize:
#
# f(x,y,z) =
#
#       1 - cos(
#           2*pi /
#           (1 + exp(-(x-125)^2 - (y-1.27)^2))
#       )
#       -----------------------------------------------
#                    1 + z^2
#
#
# Search space:
#
#       x, y, z ∈ [-128, +128]
#
#
# Topologies:
#
# 1) CLIPPING
#       Values outside [-128,+128] are clipped
#       to the nearest boundary.
#
# 2) TOROIDAL
#       The search space is periodic.
#
#       -128 and +128 are considered coincident.
#
#
# Speed limits:
#
#       16
#       32
#       64   <-- 1/4 of search-space width
#       128
#
#
# Maximum number of iterations:
#
#       10000
#
# ============================================================


import random
import math
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

RSEED = 42

# Maximum number of iterations
GEN = 10000

# Population / swarm size
SWARM_SIZE = 50

# Search domain
PMIN = -128.0
PMAX = 128.0

# Width of the search space
SPACE_WIDTH = PMAX - PMIN

# Required speed limit = 1/4 of the search-space width
REQUIRED_SPEED_LIMIT = SPACE_WIDTH / 4.0

# Variables: x, y, z
PSIZE = 3


# ============================================================
# DEAP CLASSES
# ============================================================

# Maximization
if not hasattr(creator, "FitnessMaxPSO2"):

    creator.create(
        "FitnessMaxPSO2",
        base.Fitness,
        weights=(1.0,)
    )


if not hasattr(creator, "ParticlePSO2"):

    creator.create(
        "ParticlePSO2",
        list,
        fitness=creator.FitnessMaxPSO2,
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
    # Exponent:
    #
    # -(x-125)^2 - (y-1.27)^2
    # --------------------------------------------------------

    exponent = (
        -(x - 125.0) ** 2
        -(y - 1.27) ** 2
    )

    # Prevent numerical problems
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
    speed_limit
):

    particle = creator.ParticlePSO2(
        random.uniform(pmin, pmax)
        for _ in range(size)
    )

    particle.speed = [
        random.uniform(
            -speed_limit,
            speed_limit
        )
        for _ in range(size)
    ]

    particle.smin = -speed_limit
    particle.smax = speed_limit

    particle.pmin = pmin
    particle.pmax = pmax

    return particle


# ============================================================
# TOROIDAL POSITION
# ============================================================

def wrap_position(value):

    """
    Convert a position into [-128,128)
    using a toroidal topology.

    Examples:

        128 + 10  -> -118
        128 + 50  ->  -78

        -128 - 10 ->  118
        -128 - 50 ->   78
    """

    return (
        (value - PMIN)
        % SPACE_WIDTH
    ) + PMIN


# ============================================================
# CLIPPING POSITION
# ============================================================

def clip_position(value):

    """
    Keep the position inside [-128,+128].
    """

    if value < PMIN:
        return PMIN

    if value > PMAX:
        return PMAX

    return value


# ============================================================
# PARTICLE UPDATE
# ============================================================

def update_particle(
    particle,
    global_best,
    w,
    c1,
    c2,
    topology
):

    """
    Standard PSO update:

        v(t+1) =
            w * v(t)
            + c1*r1*(pbest-x)
            + c2*r2*(gbest-x)

        x(t+1) = x(t) + v(t+1)

    Then the position is handled according to
    the selected topology.
    """

    for i in range(len(particle)):

        # Random numbers
        r1 = random.random()
        r2 = random.random()

        # ----------------------------------------------------
        # Cognitive component
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
        # ----------------------------------------------------

        social = (
            c2
            * r2
            * (
                global_best[i]
                - particle[i]
            )
        )

        # ----------------------------------------------------
        # Velocity update
        # ----------------------------------------------------

        new_speed = (
            w * particle.speed[i]
            + cognitive
            + social
        )

        # ----------------------------------------------------
        # Speed limit
        # ----------------------------------------------------

        if new_speed > particle.smax:

            new_speed = particle.smax

        elif new_speed < particle.smin:

            new_speed = particle.smin

        particle.speed[i] = new_speed

        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        new_position = (
            particle[i]
            + particle.speed[i]
        )

        # ----------------------------------------------------
        # Topology
        # ----------------------------------------------------

        if topology == "clipping":

            particle[i] = clip_position(
                new_position
            )

        elif topology == "toroidal":

            particle[i] = wrap_position(
                new_position
            )

        else:

            raise ValueError(
                "Unknown topology: "
                + str(topology)
            )


# ============================================================
# ONE PSO RUN
# ============================================================

def run_pso(
    swarm_size,
    speed_limit,
    topology,
    w,
    c1,
    c2,
    generations=GEN,
    seed=RSEED,
    verbose=False
):

    # --------------------------------------------------------
    # Random seed
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
        speed_limit=speed_limit
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
    # Create swarm
    # --------------------------------------------------------

    population = toolbox.population(
        n=swarm_size
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

    global_best = None

    # Generation in which the best solution was found
    best_generation = 0

    # --------------------------------------------------------
    # PSO LOOP
    # --------------------------------------------------------

    for generation in range(generations):

        # ====================================================
        # Evaluate particles
        # ====================================================

        for particle in population:

            particle.fitness.values = (
                toolbox.evaluate(
                    particle
                )
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
                    creator.ParticlePSO2(
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
                    creator.ParticlePSO2(
                        particle
                    )
                )

                global_best.fitness.values = (
                    particle.fitness.values
                )

                best_generation = generation

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
        # Progress
        # ====================================================

        if verbose:

            if (
                generation % 500 == 0
                or generation == generations - 1
            ):

                print(
                    f"Generation "
                    f"{generation:5d} | "
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
                c2,
                topology
            )

    return (
        global_best,
        logbook,
        best_generation
    )


# ============================================================
# PARAMETER SEARCH
# ============================================================

def parameter_search(
    swarm_size,
    speed_limit,
    topology,
    w_values,
    c1_values,
    c2_values,
    generations=GEN,
    seed=RSEED
):

    results = []

    total = (
        len(w_values)
        * len(c1_values)
        * len(c2_values)
    )

    counter = 0

    print()
    print("=" * 75)
    print(
        f"PARAMETER SEARCH"
    )
    print(
        f"Topology = {topology}"
    )
    print(
        f"Particles = {swarm_size}"
    )
    print(
        f"Speed limit = {speed_limit}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Try all combinations
    # --------------------------------------------------------

    for w in w_values:

        for c1 in c1_values:

            for c2 in c2_values:

                counter += 1

                print(
                    f"[{counter}/{total}] "
                    f"w={w:.2f}, "
                    f"c1={c1:.2f}, "
                    f"c2={c2:.2f}"
                )

                start = time.time()

                best, logbook, best_generation = (
                    run_pso(

                        swarm_size=swarm_size,

                        speed_limit=speed_limit,

                        topology=topology,

                        w=w,

                        c1=c1,

                        c2=c2,

                        generations=generations,

                        seed=seed,

                        verbose=False
                    )
                )

                elapsed = (
                    time.time() - start
                )

                fitness = (
                    best.fitness.values[0]
                )

                results.append({

                    "topology":
                    topology,

                    "population":
                    swarm_size,

                    "speed_limit":
                    speed_limit,

                    "w":
                    w,

                    "c1":
                    c1,

                    "c2":
                    c2,

                    "best_fitness":
                    fitness,

                    "best_generation":
                    best_generation,

                    "x":
                    best[0],

                    "y":
                    best[1],

                    "z":
                    best[2],

                    "time":
                    elapsed
                })

                print(
                    f"    fitness="
                    f"{fitness:.12f}"
                )

    # Highest fitness first
    results.sort(
        key=lambda result:
        result["best_fitness"],
        reverse=True
    )

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    filename,
    results
):

    if not results:
        return

    fieldnames = [
        "topology",
        "population",
        "speed_limit",
        "w",
        "c1",
        "c2",
        "best_fitness",
        "best_generation",
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
        f"\nSaved: {filename}"
    )


# ============================================================
# PRINT BEST RESULTS
# ============================================================

def print_best(
    results,
    title,
    number=10
):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    for index, result in enumerate(
        results[:number],
        start=1
    ):

        print(
            f"{index:2d}. "
            f"fitness="
            f"{result['best_fitness']:.12f} | "
            f"w={result['w']:.2f} | "
            f"c1={result['c1']:.2f} | "
            f"c2={result['c2']:.2f} | "
            f"speed={result['speed_limit']:.0f} | "
            f"x={result['x']:.6f} | "
            f"y={result['y']:.6f} | "
            f"z={result['z']:.6f}"
        )


# ============================================================
# CONVERGENCE PLOT
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
# SPEED LIMIT COMPARISON
# ============================================================

def speed_limit_experiment(
    topology,
    speed_limits,
    w,
    c1,
    c2,
    swarm_size=50,
    generations=GEN,
    seed=RSEED
):

    results = []

    print()
    print("=" * 75)
    print(
        f"SPEED LIMIT COMPARISON"
    )
    print(
        f"Topology = {topology}"
    )
    print("=" * 75)

    for speed_limit in speed_limits:

        print(
            f"\nSpeed limit = "
            f"{speed_limit}"
        )

        best, logbook, best_generation = (
            run_pso(

                swarm_size=swarm_size,

                speed_limit=speed_limit,

                topology=topology,

                w=w,

                c1=c1,

                c2=c2,

                generations=generations,

                seed=seed,

                verbose=False
            )
        )

        fitness = (
            best.fitness.values[0]
        )

        result = {

            "topology":
            topology,

            "population":
            swarm_size,

            "speed_limit":
            speed_limit,

            "w":
            w,

            "c1":
            c1,

            "c2":
            c2,

            "best_fitness":
            fitness,

            "best_generation":
            best_generation,

            "x":
            best[0],

            "y":
            best[1],

            "z":
            best[2]
        }

        results.append(result)

        print(
            f"Best fitness = "
            f"{fitness:.12f}"
        )

    return results


# ============================================================
# TOPOLOGY COMPARISON PLOT
# ============================================================

def plot_topology_comparison(
    log_clipping,
    log_toroidal,
    speed_limit
):

    gen1 = log_clipping.select(
        "gen"
    )

    best1 = log_clipping.select(
        "max"
    )

    gen2 = log_toroidal.select(
        "gen"
    )

    best2 = log_toroidal.select(
        "max"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        gen1,
        best1,
        label="Clipping"
    )

    plt.plot(
        gen2,
        best2,
        label="Toroidal"
    )

    plt.xlabel(
        "Generation"
    )

    plt.ylabel(
        "Best fitness"
    )

    plt.title(
        "Clipping vs Toroidal topology\n"
        f"Speed limit = {speed_limit}"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "pso_topology_comparison.png",
        dpi=150
    )

    plt.show()


# ============================================================
# SPEED LIMIT PLOT
# ============================================================

def plot_speed_limits(
    results,
    topology
):

    speed_limits = [
        r["speed_limit"]
        for r in results
    ]

    fitness = [
        r["best_fitness"]
        for r in results
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        speed_limits,
        fitness,
        marker="o"
    )

    plt.xlabel(
        "Speed limit"
    )

    plt.ylabel(
        "Best fitness"
    )

    plt.title(
        f"Effect of speed limit - "
        f"{topology}"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    filename = (
        "pso_speed_limits_"
        + topology
        + ".png"
    )

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
        f"\nSearch space:"
        f" [{PMIN}, {PMAX}]"
    )

    print(
        f"Search-space width:"
        f" {SPACE_WIDTH}"
    )

    print(
        f"Required speed limit:"
        f" {REQUIRED_SPEED_LIMIT}"
    )

    print(
        f"Maximum iterations:"
        f" {GEN}"
    )


    # ========================================================
    # PARAMETER VALUES
    # ========================================================
    #
    # All values respect the required ranges:
    #
    # w  ∈ [0,1]
    # c1 ∈ [0,2]
    # c2 ∈ [0,2]
    #
    # --------------------------------------------------------
    # You can make this grid finer later.
    # ========================================================

    w_values = [
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


    # ========================================================
    # SPEED LIMITS
    # ========================================================

    speed_limits = [
        16,
        32,
        64,
        128
    ]


    # ========================================================
    # STEP 1
    #
    # Find good w, c1, c2 with 50 particles.
    #
    # First use the REQUIRED speed limit = 64.
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STEP 1 - PARAMETER SEARCH"
    )
    print(
        "50 particles, speed limit = 64"
    )
    print("=" * 75)

    results_50_clipping = parameter_search(

        swarm_size=50,

        speed_limit=64,

        topology="clipping",

        w_values=w_values,

        c1_values=c1_values,

        c2_values=c2_values,

        generations=GEN,

        seed=RSEED
    )

    print_best(
        results_50_clipping,
        "BEST PARAMETERS - 50 PARTICLES - CLIPPING"
    )

    save_results(
        "results_50_clipping.csv",
        results_50_clipping
    )


    # ========================================================
    # STEP 1B
    #
    # Do the same with toroidal topology.
    # ========================================================

    results_50_toroidal = parameter_search(

        swarm_size=50,

        speed_limit=64,

        topology="toroidal",

        w_values=w_values,

        c1_values=c1_values,

        c2_values=c2_values,

        generations=GEN,

        seed=RSEED
    )

    print_best(
        results_50_toroidal,
        "BEST PARAMETERS - 50 PARTICLES - TOROIDAL"
    )

    save_results(
        "results_50_toroidal.csv",
        results_50_toroidal
    )


    # ========================================================
    # Select the best parameter set found with 50 particles.
    #
    # We compare both topologies.
    # ========================================================

    best_clipping = (
        results_50_clipping[0]
    )

    best_toroidal = (
        results_50_toroidal[0]
    )

    if (
        best_toroidal["best_fitness"]
        >
        best_clipping["best_fitness"]
    ):

        best_50 = best_toroidal

    else:

        best_50 = best_clipping


    print()
    print("=" * 75)
    print(
        "BEST CONFIGURATION WITH 50 PARTICLES"
    )
    print("=" * 75)

    print(
        f"Topology = "
        f"{best_50['topology']}"
    )

    print(
        f"w = "
        f"{best_50['w']}"
    )

    print(
        f"c1 = "
        f"{best_50['c1']}"
    )

    print(
        f"c2 = "
        f"{best_50['c2']}"
    )

    print(
        f"Speed limit = "
        f"{best_50['speed_limit']}"
    )

    print(
        f"Fitness = "
        f"{best_50['best_fitness']:.12f}"
    )


    # ========================================================
    # STEP 2
    #
    # 100 particles
    #
    # Use the best w, c1, c2 from step 1.
    #
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STEP 2 - 100 PARTICLES"
    )
    print(
        "Using best parameters from step 1"
    )
    print("=" * 75)

    best_100_step2, log_100_step2, generation_100_step2 = (
        run_pso(

            swarm_size=100,

            speed_limit=64,

            topology=best_50["topology"],

            w=best_50["w"],

            c1=best_50["c1"],

            c2=best_50["c2"],

            generations=GEN,

            seed=RSEED,

            verbose=True
        )
    )

    fitness_100_step2 = (
        best_100_step2
        .fitness
        .values[0]
    )

    print()
    print(
        "STEP 2 RESULT"
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
        f"fitness = "
        f"{fitness_100_step2:.12f}"
    )

    print(
        f"best generation = "
        f"{generation_100_step2}"
    )


    plot_convergence(

        log_100_step2,

        (
            "PSO - 100 particles - "
            "parameters from step 1"
        ),

        "pso_step2_convergence.png"
    )


    # ========================================================
    # STEP 3
    #
    # 100 particles.
    #
    # Search again for better parameters.
    # ========================================================

    print()
    print("=" * 75)
    print(
        "STEP 3 - RE-OPTIMIZATION"
    )
    print(
        "100 particles - search for better parameters"
    )
    print("=" * 75)

    results_100 = parameter_search(

        swarm_size=100,

        speed_limit=64,

        topology=best_50["topology"],

        w_values=w_values,

        c1_values=c1_values,

        c2_values=c2_values,

        generations=GEN,

        seed=RSEED
    )

    print_best(
        results_100,
        "BEST PARAMETERS - 100 PARTICLES"
    )

    save_results(
        "results_100.csv",
        results_100
    )

    best_100 = results_100[0]


    # ========================================================
    # SPEED LIMIT EXPERIMENT
    #
    # Compare different speed limits.
    #
    # Use the best parameters found with 100 particles.
    # ========================================================

    print()
    print("=" * 75)
    print(
        "SPEED LIMIT EXPERIMENT"
    )
    print("=" * 75)

    speed_results = speed_limit_experiment(

        topology=best_50["topology"],

        speed_limits=speed_limits,

        w=best_100["w"],

        c1=best_100["c1"],

        c2=best_100["c2"],

        swarm_size=100,

        generations=GEN,

        seed=RSEED
    )

    save_results(
        "speed_limit_results.csv",
        speed_results
    )

    plot_speed_limits(
        speed_results,
        best_50["topology"]
    )


    # ========================================================
    # TOPOLOGY COMPARISON
    #
    # Use the required speed limit = 64 and the best
    # parameters found for 50 particles.
    # ========================================================

    print()
    print("=" * 75)
    print(
        "TOPOLOGY COMPARISON"
    )
    print(
        "Clipping vs Toroidal"
    )
    print("=" * 75)

    best_clip_particle, log_clip, gen_clip = (
        run_pso(

            swarm_size=50,

            speed_limit=64,

            topology="clipping",

            w=best_clipping["w"],

            c1=best_clipping["c1"],

            c2=best_clipping["c2"],

            generations=GEN,

            seed=RSEED,

            verbose=False
        )
    )

    best_tor_particle, log_tor, gen_tor = (
        run_pso(

            swarm_size=50,

            speed_limit=64,

            topology="toroidal",

            w=best_toroidal["w"],

            c1=best_toroidal["c1"],

            c2=best_toroidal["c2"],

            generations=GEN,

            seed=RSEED,

            verbose=False
        )
    )

    print()
    print(
        f"Clipping fitness = "
        f"{best_clip_particle.fitness.values[0]:.12f}"
    )

    print(
        f"Toroidal fitness = "
        f"{best_tor_particle.fitness.values[0]:.12f}"
    )

    plot_topology_comparison(

        log_clip,

        log_tor,

        speed_limit=64
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 75)

    print()
    print(
        "50 particles - best configuration:"
    )

    print(
        f"    topology = "
        f"{best_50['topology']}"
    )

    print(
        f"    w = "
        f"{best_50['w']}"
    )

    print(
        f"    c1 = "
        f"{best_50['c1']}"
    )

    print(
        f"    c2 = "
        f"{best_50['c2']}"
    )

    print(
        f"    fitness = "
        f"{best_50['best_fitness']:.12f}"
    )

    print()
    print(
        "100 particles - same parameters:"
    )

    print(
        f"    fitness = "
        f"{fitness_100_step2:.12f}"
    )

    print()
    print(
        "100 particles - re-optimized:"
    )

    print(
        f"    topology = "
        f"{best_100['topology']}"
    )

    print(
        f"    w = "
        f"{best_100['w']}"
    )

    print(
        f"    c1 = "
        f"{best_100['c1']}"
    )

    print(
        f"    c2 = "
        f"{best_100['c2']}"
    )

    print(
        f"    fitness = "
        f"{best_100['best_fitness']:.12f}"
    )

    print()
    print(
        "Required speed limit:"
    )

    print(
        f"    {REQUIRED_SPEED_LIMIT}"
    )

    print()
    print(
        "Theoretical upper bound:"
    )

    print(
        "    f(x,y,z) <= 2"
    )

    print()
    print(
        "Experiments completed."
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()