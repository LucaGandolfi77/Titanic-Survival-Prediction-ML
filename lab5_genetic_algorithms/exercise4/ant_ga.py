"""Exercise 4 — GA that Evolves a Sequence of Moves for the Ant Game.

Individual: list of 40 bits (20 moves × 2 bits each).
Fitness: maximize score from run_game (number of food cells eaten).

NOTE: This evolves the best move sequence for ONE fixed board instance,
not a general strategy. The board is generated once and held constant.
"""

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exercise4.ant_game import generate_board, run_game, decode_moves, MOVE_NAMES
from utils.plotting import plot_convergence_single, plot_ant_game


# GA parameters
N_MOVES = 20
BITS_PER_MOVE = 2
TOTAL_BITS = N_MOVES * BITS_PER_MOVE
POP_SIZE = 200
NGEN = 200
CXPB = 0.7
MUTPB = 0.3


def make_eval_func(board):
    """Create a fitness function for a given fixed board.

    Args:
        board: 2D numpy array (NxN) with 1=food, 0=empty.

    Returns:
        Evaluation function compatible with DEAP.
    """
    def eval_ant(individual):
        """Evaluate the ant's move sequence on the fixed board.

        Args:
            individual: list of 40 bits.

        Returns:
            Tuple with the score (number of food cells eaten).
        """
        moves = decode_moves(individual)
        score, _ = run_game(board, moves)
        return (score,)

    return eval_ant


def run(verbose=True):
    """Run the Ant Game GA for Exercise 4.

    Args:
        verbose: if True, print progress info.

    Returns:
        Tuple of (best_moves, best_score, logbook, board).
    """
    random.seed(42)
    np.random.seed(42)

    # Generate the fixed board
    board = generate_board(n=10, food_ratio=0.3, seed=42)
    total_food = int(board.sum())

    if verbose:
        print("=" * 60)
        print(f"Exercise 4 — Ant Game GA (board 10x10, {total_food} food cells)")
        print("=" * 60)
        print(f"  Individual: {TOTAL_BITS} bits ({N_MOVES} moves × {BITS_PER_MOVE} bits)")
        print(f"  Population: {POP_SIZE}, Generations: {NGEN}")
        print(f"  Pc={CXPB}, Pm={MUTPB}")
        print()

    # --- DEAP setup ---
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bit, n=TOTAL_BITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_func = make_eval_func(board)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / TOTAL_BITS)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Run GA ---
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                       ngen=NGEN, stats=stats, halloffame=hof,
                                       verbose=verbose)

    # --- Results ---
    best = hof[0]
    best_moves = decode_moves(best)
    best_score, best_path = run_game(board, best_moves)
    nf = POP_SIZE + NGEN * POP_SIZE

    if verbose:
        move_names = [MOVE_NAMES[m] for m in best_moves]
        print(f"\n  Best move sequence: {move_names}")
        print(f"  Best score: {best_score}/{total_food} food cells eaten")
        print(f"  Total Nf (approx): {nf}")

    # Plot convergence
    output_dir = os.path.dirname(__file__)
    plot_convergence_single(logbook, "Exercise 4 — Ant Game Convergence",
                            os.path.join(output_dir, "exercise4_convergence.png"),
                            best_key="max")

    # Visualize game
    plot_ant_game(board, best_path,
                  f"Ant Game — Best Path (score={best_score}/{total_food})",
                  os.path.join(output_dir, "exercise4_ant_path.png"))

    return best_moves, best_score, logbook, board


if __name__ == "__main__":
    run(verbose=True)
