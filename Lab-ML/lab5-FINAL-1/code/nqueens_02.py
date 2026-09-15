import random
import csv
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


# Valore iniziale. Verrà cambiato nel ciclo finale.
NB_QUEENS = 6


def evalNQueens(individual):
    """Conta i conflitti diagonali tra le regine."""

    size = len(individual)

    left_diagonal = [0] * (2 * size - 1)
    right_diagonal = [0] * (2 * size - 1)

    # Conta quante regine si trovano su ogni diagonale
    for i in range(size):
        left_diagonal[i + individual[i]] += 1
        right_diagonal[size - 1 - i + individual[i]] += 1

    # Conta i conflitti
    sum_ = 0

    for i in range(2 * size - 1):
        if left_diagonal[i] > 1:
            sum_ += left_diagonal[i] - 1

        if right_diagonal[i] > 1:
            sum_ += right_diagonal[i] - 1

    return (sum_,)


# Evita errori se il codice viene eseguito più volte,
# per esempio in Jupyter Notebook.
if not hasattr(creator, "FitnessMin"):
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin
    )


toolbox = base.Toolbox()

toolbox.register("evaluate", evalNQueens)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("select", tools.selTournament, tournsize=3)


def configure_toolbox(number_of_queens):
    """
    Riconfigura la creazione degli individui e la mutazione
    quando cambia il numero di regine.
    """

    toolbox.register(
        "permutation",
        random.sample,
        range(number_of_queens),
        number_of_queens
    )

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.permutation
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    toolbox.register(
        "mutate",
        tools.mutShuffleIndexes,
        indpb=2.0 / number_of_queens
    )


def main(number_of_queens, seed=0):
    random.seed(seed)

    # Aggiorna il toolbox per il numero corrente di regine
    configure_toolbox(number_of_queens)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(
        lambda ind: ind.fitness.values
    )

    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    # Salviamo sia popolazione sia logbook
    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    return pop, log, hof


if __name__ == "__main__":

    NUMBER_OF_RUNS = 10

    final_results = []

    # Prova NB_QUEENS da 2 a 50 inclusi
    for number_of_queens in range(2, 51):

        successful_runs = 0
        run_results = []

        print("\n" + "=" * 60)
        print(f"NB_QUEENS = {number_of_queens}")
        print("=" * 60)

        # Esegue 10 run con seed da 0 a 9
        for seed in range(NUMBER_OF_RUNS):

            pop, log, hof = main(
                number_of_queens=number_of_queens,
                seed=seed
            )

            # La generazione 100 è l'ultima riga del logbook
            min_at_generation_100 = log[-1]["Min"]

            # Miglior fitness trovato nell'intera run
            best_fitness = hof[0].fitness.values[0]

            success = min_at_generation_100 == 0

            if success:
                successful_runs += 1

            run_results.append(min_at_generation_100)

            print(
                f"Run {seed + 1:2d}/10 | "
                f"Seed: {seed} | "
                f"Min generation 100: "
                f"{min_at_generation_100:.0f} | "
                f"Best overall: {best_fitness:.0f} | "
                f"{'SUCCESS' if success else 'FAIL'}"
            )

        # Calcolo della percentuale
        success_percentage = (
            successful_runs / NUMBER_OF_RUNS
        ) * 100

        final_results.append({
            "NB_QUEENS": number_of_queens,
            "successful_runs": successful_runs,
            "total_runs": NUMBER_OF_RUNS,
            "success_percentage": success_percentage,
            "run_results": run_results
        })

        print(
            f"\nResult for {number_of_queens} queens: "
            f"{successful_runs}/{NUMBER_OF_RUNS} successful runs "
            f"= {success_percentage:.1f}%"
        )


    # --------------------------------------------------------
    # Risultati finali
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'Queens':>8} | "
        f"{'Success':>10} | "
        f"{'Percentage':>12}"
    )

    print("-" * 38)

    for result in final_results:
        print(
            f"{result['NB_QUEENS']:>8} | "
            f"{result['successful_runs']:>7}/10 | "
            f"{result['success_percentage']:>10.1f}%"
        )


    # --------------------------------------------------------
    # Salvataggio CSV
    # --------------------------------------------------------

    with open(
        "n_queens_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "NB_QUEENS",
            "SUCCESSFUL_RUNS",
            "TOTAL_RUNS",
            "SUCCESS_PERCENTAGE",
            "RUN_1_MIN",
            "RUN_2_MIN",
            "RUN_3_MIN",
            "RUN_4_MIN",
            "RUN_5_MIN",
            "RUN_6_MIN",
            "RUN_7_MIN",
            "RUN_8_MIN",
            "RUN_9_MIN",
            "RUN_10_MIN"
        ])

        for result in final_results:
            writer.writerow([
                result["NB_QUEENS"],
                result["successful_runs"],
                result["total_runs"],
                result["success_percentage"],
                *result["run_results"]
            ])

    print("\nResults saved to: n_queens_results.csv")