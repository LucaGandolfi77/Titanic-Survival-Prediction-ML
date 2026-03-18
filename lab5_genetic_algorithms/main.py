#!/usr/bin/env python3
"""Main entry point for Lab 5 — Genetic Algorithms.

Run specific exercises or all of them via command-line arguments.

Usage:
    python main.py --exercise 1   # Exercise 1 (float + binary optimization)
    python main.py --exercise 2   # Exercise 2 (pattern guessing + scaling)
    python main.py --exercise 3   # Exercise 3 (N-Queens smart + dumb)
    python main.py --exercise 4   # Exercise 4 (Ant game)
    python main.py --all          # Run everything
"""

import argparse
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_exercise1():
    """Run Exercise 1: Function Optimization (float + binary representations)."""
    from exercise1 import float_repr, binary_repr
    from utils.plotting import plot_convergence

    print("\n" + "#" * 70)
    print("#  EXERCISE 1 — Function Optimization")
    print("#" * 70 + "\n")

    # Run both versions
    best_float, fit_float, logbook_float, hof_float = float_repr.run(verbose=True)
    print()
    best_bin, fit_bin, logbook_bin, hof_bin = binary_repr.run(verbose=True)

    # Combined convergence plot
    output_dir = os.path.join(os.path.dirname(__file__), "exercise1")
    plot_convergence(
        [logbook_float, logbook_bin],
        ["Float Representation", "Binary Representation"],
        "Exercise 1 — Convergence Comparison",
        os.path.join(output_dir, "exercise1_convergence.png"),
    )

    print("\n" + "-" * 60)
    print("EXERCISE 1 SUMMARY")
    print("-" * 60)
    print(f"  Float: best fitness = {fit_float:.6f}, "
          f"solution = ({best_float[0]:.4f}, {best_float[1]:.4f}, {best_float[2]:.4f})")
    print(f"  Binary: best fitness = {fit_bin:.6f}, "
          f"solution = ({best_bin[0]:.4f}, {best_bin[1]:.4f}, {best_bin[2]:.4f})")


def run_exercise2():
    """Run Exercise 2: Pattern Guessing + Scaling Experiment."""
    from exercise2 import pattern_guesser, scaling_experiment

    print("\n" + "#" * 70)
    print("#  EXERCISE 2 — Pattern Guessing")
    print("#" * 70 + "\n")

    # Part A: Pattern guessing
    best, best_fit, logbook, nf_total, rows, cols = pattern_guesser.run(verbose=True)

    print("\n" + "-" * 60)
    print("EXERCISE 2A SUMMARY")
    print("-" * 60)
    print(f"  Best fitness: {best_fit}/{rows * cols}")
    print(f"  Total Nf: {nf_total}")

    # Part B: Scaling experiment
    print()
    results = scaling_experiment.run(verbose=True)

    print("\n" + "-" * 60)
    print("EXERCISE 2B SUMMARY")
    print("-" * 60)
    print(f"  Best hyperparameters: pop_size={results['hyperparams'][0]}, "
          f"ngen={results['hyperparams'][1]}")


def run_exercise3():
    """Run Exercise 3: N-Queens (smart + dumb representations)."""
    from exercise3 import nqueens_smart, nqueens_dumb

    print("\n" + "#" * 70)
    print("#  EXERCISE 3 — N-Queens Problem")
    print("#" * 70 + "\n")

    smart_results = nqueens_smart.run(verbose=True)
    print()
    dumb_results = nqueens_dumb.run(verbose=True)

    # Comparison table
    nqueens_dumb.print_comparison(smart_results, dumb_results, verbose=True)


def run_exercise4():
    """Run Exercise 4: Ant Game."""
    from exercise4 import ant_ga

    print("\n" + "#" * 70)
    print("#  EXERCISE 4 — Ant Game")
    print("#" * 70 + "\n")

    best_moves, best_score, logbook, board = ant_ga.run(verbose=True)

    print("\n" + "-" * 60)
    print("EXERCISE 4 SUMMARY")
    print("-" * 60)
    print(f"  Best score: {best_score}/{int(board.sum())} food cells")
    print(f"  Move sequence length: {len(best_moves)}")


def main():
    """Parse arguments and run the requested exercises."""
    parser = argparse.ArgumentParser(
        description="Lab 5 — Genetic Algorithms with DEAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exercise", type=int, choices=[1, 2, 3, 4],
                        help="Run a specific exercise (1-4)")
    parser.add_argument("--all", action="store_true",
                        help="Run all exercises")

    args = parser.parse_args()

    if not args.exercise and not args.all:
        parser.print_help()
        sys.exit(1)

    runners = {
        1: run_exercise1,
        2: run_exercise2,
        3: run_exercise3,
        4: run_exercise4,
    }

    if args.all:
        for ex_num in sorted(runners):
            runners[ex_num]()
    else:
        runners[args.exercise]()

    print("\n" + "=" * 70)
    print("Done! Check the exercise folders for generated PNG plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
