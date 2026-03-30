"""
Run multi-objective NSGA-II experiment across all datasets and seeds.

Usage:
    python -m experiments.run_multi_objective
    python -m experiments.run_multi_objective --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from evolutionary_automl.config import CFG
from evolutionary_automl.evolution.multi_objective import run_nsga2
from evolutionary_automl.visualization.fitness_curves import plot_fitness_evolution
from evolutionary_automl.visualization.pareto_front import (
    plot_pareto_front_matplotlib,
    plot_pareto_front_plotly,
)

logger = logging.getLogger(__name__)

DATASET_LOADERS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def run_experiment(dry_run: bool = False) -> dict:
    """Run NSGA-II across datasets and seeds.

    Args:
        dry_run: If True, run 2 generations on iris only.

    Returns:
        Full results dictionary.
    """
    CFG.ensure_dirs()

    datasets = ["iris"] if dry_run else list(CFG.DATASET_NAMES)
    seeds = CFG.RANDOM_SEEDS[:2] if dry_run else CFG.RANDOM_SEEDS[:CFG.N_RUNS]
    pop_size = 20 if dry_run else CFG.POP_SIZE_NSGA
    n_gen = 2 if dry_run else CFG.N_GEN_NSGA

    all_results = {}
    total_start = time.perf_counter()

    for ds_name in datasets:
        logger.info(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        X, y = DATASET_LOADERS[ds_name](return_X_y=True)
        ds_results = {"seeds": {}, "best_f1_scores": []}

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")
            result = run_nsga2(
                X, y,
                dataset_name=ds_name,
                pop_size=pop_size,
                n_gen=n_gen,
                seed=seed,
            )
            ds_results["seeds"][seed] = result
            if result["best_f1_individual"]:
                ds_results["best_f1_scores"].append(
                    result["best_f1_individual"]["f1"]
                )

            if seed == seeds[0]:
                plot_fitness_evolution(
                    result["history"],
                    title=f"NSGA-II Fitness Evolution — {ds_name}",
                    save_path=CFG.PLOTS_DIR / f"nsga2_fitness_{ds_name}.png",
                )
                if result["pareto_front"]:
                    plot_pareto_front_matplotlib(
                        result["pareto_front"],
                        title=f"Pareto Front — {ds_name}",
                        save_path=CFG.PLOTS_DIR / f"pareto_front_{ds_name}.png",
                    )
                    plot_pareto_front_plotly(
                        result["pareto_front"],
                        title=f"Pareto Front — {ds_name}",
                        save_path_html=CFG.PLOTS_DIR / f"pareto_front_{ds_name}.html",
                    )
            plt.close("all")

        f1_arr = np.array(ds_results["best_f1_scores"]) if ds_results["best_f1_scores"] else np.array([0.0])
        ds_results["summary"] = {
            "mean_f1": float(np.mean(f1_arr)),
            "std_f1": float(np.std(f1_arr, ddof=1)) if len(f1_arr) > 1 else 0.0,
            "median_f1": float(np.median(f1_arr)),
        }
        logger.info(
            f"\n{ds_name} NSGA-II Summary: "
            f"F1 = {ds_results['summary']['mean_f1']:.4f} ± "
            f"{ds_results['summary']['std_f1']:.4f}"
        )
        all_results[ds_name] = ds_results

    total_time = time.perf_counter() - total_start
    all_results["_meta"] = {
        "method": "nsga2",
        "total_time_seconds": total_time,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "n_runs": len(seeds),
        "dry_run": dry_run,
    }

    output_path = CFG.RESULTS_DIR / "multi_objective_results.json"

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    output_path.write_text(json.dumps(all_results, indent=2, default=_serialize))
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total wall clock time: {total_time:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run NSGA-II experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 2 generations on iris only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_experiment(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
