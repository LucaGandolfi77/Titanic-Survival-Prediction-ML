# ecosim/main.py
from __future__ import annotations
import argparse
import sys
from .config import SimulationConfig
from .simulation import Simulation
from .logging_utils import log_step, write_csv, plot_results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grass‑herbivore‑carnivore ecosystem simulation."
    )
    parser.add_argument(
        "--num-steps", type=int, help="Number of time steps to simulate."
    )
    parser.add_argument(
        "--random-seed", type=int, help="Seed for random number generator."
    )
    parser.add_argument(
        "--preset",
        choices=["balanced", "high_grass", "predator_heavy", "low_regrowth"],
        default="balanced",
        help="Configuration preset.",
    )
    parser.add_argument(
        "--log-file",
        default="ecosim_log.csv",
        help="CSV file to write detailed logs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate a population plot.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per‑step console output.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = SimulationConfig.preset(args.preset)
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.random_seed is not None:
        config.random_seed = args.random_seed

    sim = Simulation(config)
    results = sim.run()

    # Logging
    for step_data in results:
        log_step(step_data, verbose=not args.quiet)

    write_csv(results, args.log_file)
    if not args.no_plot:
        plot_results(args.log_file)

    # Final summary
    final = results[-1] if results else {}
    print("\n=== Simulation Summary ===")
    print(f"Steps run: {config.num_steps}")
    print(f"Final grass: {final.get('grass', 0):.1f} "
          f"({final.get('grass_fraction', 0):.2%} of carrying capacity)")
    print(f"Final herbivores: {final.get('herbivores', 0)}")
    print(f"Final carnivores: {final.get('carnivores', 0)}")
    # Detect functional extinction (population zero for last 10% of steps)
    if results:
        tail_len = max(1, config.num_steps // 10)
        tail = results[-tail_len:]
        herb_extinct = all(r["herbivores"] == 0 for r in tail)
        carn_extinct = all(r["carnivores"] == 0 for r in tail)
        if herb_extinct:
            print("Warning: Herbivores functionally extinct in final tail.")
        if carn_extinct:
            print("Warning: Carnivores functionally extinct in final tail.")

if __name__ == "__main__":
    main()
