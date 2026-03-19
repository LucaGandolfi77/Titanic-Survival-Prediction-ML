#!/usr/bin/env python3
"""
main.py — CLI entry point for the co-evolutionary predator-prey simulation.

Usage examples:
    python main.py                           # run DEFAULT preset
    python main.py --preset BLIND_PREY       # run BLIND_PREY preset
    python main.py --generations 50 --seed 7 # custom params
    python main.py --animate                 # also produce a GIF of the final episode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import SimConfig, get_preset, PRESETS
from coevolution import CoevolutionEngine
from behavior_analysis import analyze_trajectories_labels
from logging_utils import CSVLogger, print_generation, save_best_trees
from visualization import (
    plot_fitness_curves,
    plot_behavior_emergence,
    animate_episode,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse CLI arguments.

    Args:
        argv: Optional explicit arg list (for testing). Uses sys.argv if None.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Co-evolutionary predator-prey simulation with Genetic Programming.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="DEFAULT",
        choices=[k for k in PRESETS],
        help="Named configuration preset (default: DEFAULT).",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override number of generations (default: from preset).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override master random seed (default: from preset).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: from preset).",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Generate a GIF animation of the final episode.",
    )
    p.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Override GP population size.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the simulation end-to-end.

    Args:
        argv: Optional arg list for testing.
    """
    args = parse_args(argv)

    # ---- Build config -------------------------------------------------------
    cfg: SimConfig = get_preset(args.preset)
    if args.generations is not None:
        cfg.generations = args.generations
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.pop_size is not None:
        cfg.pop_size = args.pop_size

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Co-evolutionary Predator-Prey Simulation ===")
    print(f"Preset : {args.preset}")
    print(f"Grid   : {cfg.grid_size}x{cfg.grid_size}")
    print(f"Prey obs radius    : {cfg.prey_obs_radius}")
    print(f"Predator obs radius: {cfg.predator_obs_radius}")
    print(f"Populations        : {cfg.pop_size}")
    print(f"Generations        : {cfg.generations}")
    print(f"Seed               : {cfg.seed}")
    print(f"Output dir         : {cfg.output_dir}")
    print()

    # ---- Create engine ------------------------------------------------------
    engine = CoevolutionEngine(cfg)

    # ---- CSV logger ---------------------------------------------------------
    csv_logger = CSVLogger(out_dir)
    csv_logger.open()

    gen_counter = [0]  # mutable counter for the log_callback closure

    def log_callback(stats):
        gen = stats.generation
        print_generation(gen, cfg.generations, stats)
        csv_logger.log(gen, stats)

        # Save best trees periodically.
        if cfg.log_tree_every > 0 and (gen % cfg.log_tree_every == 0 or gen == cfg.generations - 1):
            best_prey = engine.prey_pop[0]  # selBest already happened inside engine
            best_pred = engine.pred_pop[0]
            from gp_setup import get_prey_pset, get_predator_pset
            save_best_trees(
                out_dir, gen,
                best_prey, best_pred,
                get_prey_pset(cfg), get_predator_pset(cfg),
            )

    # ---- Run ----------------------------------------------------------------
    t0 = time.perf_counter()

    result = engine.run(
        behavior_callback=analyze_trajectories_labels,
        log_callback=log_callback,
    )

    elapsed = time.perf_counter() - t0
    csv_logger.close()

    # ---- Summary ------------------------------------------------------------
    print()
    print(f"=== Finished in {elapsed:.1f}s ===")

    if result.generation_stats:
        last = result.generation_stats[-1]
        print(f"Final prey fitness  : mean={last.prey_fitness_mean:.2f}  best={last.prey_fitness_best:.2f}")
        print(f"Final pred fitness  : mean={last.pred_fitness_mean:.2f}  best={last.pred_fitness_best:.2f}")
        print(f"Avg prey survival   : {last.avg_prey_survival:.1f} steps")
        print(f"Avg food per prey   : {last.avg_prey_food:.1f}")
        print(f"Avg kills per pred  : {last.avg_pred_kills:.1f}")
        if last.behaviors:
            print(f"Detected behaviours : {', '.join(last.behaviors)}")
        else:
            print("Detected behaviours : none")

    # ---- Plots --------------------------------------------------------------
    print()
    print("Generating plots...")

    fc = plot_fitness_curves(result, out_dir / "fitness_curves.png")
    print(f"  -> {fc}")

    be = plot_behavior_emergence(result, out_dir / "behavior_emergence.png")
    print(f"  -> {be}")

    if args.animate and result.sample_episodes:
        gif = animate_episode(
            result.sample_episodes[-1],
            cfg.grid_size,
            out_dir / "episode.gif",
        )
        print(f"  -> {gif}")

    print()
    print("Done. All outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
