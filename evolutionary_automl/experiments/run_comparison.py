"""
Full comparison experiment: GA vs NSGA-II vs RandomSearch vs GridSearch vs Manual.

Runs all methods on all datasets with multiple seeds, performs statistical
tests, and generates comparison plots and tables.

Usage:
    python -m experiments.run_comparison
    python -m experiments.run_comparison --dry-run
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
from sklearn.model_selection import train_test_split

from evolutionary_automl.config import CFG
from evolutionary_automl.baselines.grid_search import run_grid_search
from evolutionary_automl.baselines.manual_tuning import run_manual_tuning
from evolutionary_automl.baselines.random_search import run_random_search
from evolutionary_automl.evolution.single_objective import run_single_objective_ga
from evolutionary_automl.evolution.multi_objective import run_nsga2
from evolutionary_automl.evaluation.statistical_tests import (
    compute_summary_stats,
    friedman_test,
    wilcoxon_signed_rank,
)
from evolutionary_automl.evaluation.report_generator import generate_latex_table, generate_markdown_table
from evolutionary_automl.genome.chromosome import chromosome_to_pipeline
from evolutionary_automl.visualization.comparison_plots import (
    plot_comparison_boxplot,
    plot_convergence_comparison,
)
from evolutionary_automl.visualization.confusion_matrix import plot_confusion_matrix

logger = logging.getLogger(__name__)

DATASET_LOADERS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def run_experiment(dry_run: bool = False) -> dict:
    """Run full comparison experiment.

    Args:
        dry_run: If True, minimal configuration for pipeline verification.

    Returns:
        Complete results dictionary with statistical tests.
    """
    CFG.ensure_dirs()

    datasets = ["iris"] if dry_run else list(CFG.DATASET_NAMES)
    seeds = CFG.RANDOM_SEEDS[:2] if dry_run else CFG.RANDOM_SEEDS[:CFG.N_RUNS]
    ga_pop = 10 if dry_run else CFG.POP_SIZE_GA
    ga_gen = 2 if dry_run else CFG.N_GEN_GA
    nsga_pop = 20 if dry_run else CFG.POP_SIZE_NSGA
    nsga_gen = 2 if dry_run else CFG.N_GEN_NSGA
    rs_iter = 20 if dry_run else 200

    all_results = {}
    total_start = time.perf_counter()

    for ds_name in datasets:
        logger.info(f"\n{'='*60}\n  DATASET: {ds_name}\n{'='*60}")
        loader = DATASET_LOADERS[ds_name]
        data = loader()
        X, y = data.data, data.target
        class_names = list(data.target_names) if hasattr(data, "target_names") else None

        ds_results = {
            "GA": {"f1_scores": [], "times": []},
            "NSGA-II": {"f1_scores": [], "times": []},
            "RandomSearch": {"f1_scores": [], "times": []},
            "GridSearch": {"f1_scores": [], "times": []},
            "Manual": {"f1_scores": [], "times": []},
        }
        ga_histories = []
        nsga_histories = []
        best_ga_individual = None
        best_ga_f1 = -1.0

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")

            # GA
            ga_res = run_single_objective_ga(
                X, y, dataset_name=ds_name,
                pop_size=ga_pop, n_gen=ga_gen, seed=seed,
            )
            ds_results["GA"]["f1_scores"].append(ga_res["best_fitness"])
            ds_results["GA"]["times"].append(
                ga_res["history"][-1]["elapsed_time"] if ga_res["history"] else 0.0
            )
            ga_histories.append(ga_res["history"])
            if ga_res["best_fitness"] > best_ga_f1:
                best_ga_f1 = ga_res["best_fitness"]
                best_ga_individual = ga_res["best_individual"]

            # NSGA-II
            nsga_res = run_nsga2(
                X, y, dataset_name=ds_name,
                pop_size=nsga_pop, n_gen=nsga_gen, seed=seed,
            )
            best_nsga_f1 = nsga_res["best_f1_individual"]["f1"] if nsga_res["best_f1_individual"] else 0.0
            ds_results["NSGA-II"]["f1_scores"].append(best_nsga_f1)
            ds_results["NSGA-II"]["times"].append(
                nsga_res["history"][-1]["elapsed_time"] if nsga_res["history"] else 0.0
            )
            nsga_histories.append(nsga_res["history"])

            # Random Search
            rs_res = run_random_search(
                X, y, dataset_name=ds_name, n_iter=rs_iter, seed=seed,
            )
            ds_results["RandomSearch"]["f1_scores"].append(rs_res["best_fitness"])
            ds_results["RandomSearch"]["times"].append(rs_res["wall_clock_time"])

            # Grid Search
            gs_res = run_grid_search(X, y, dataset_name=ds_name, seed=seed)
            ds_results["GridSearch"]["f1_scores"].append(gs_res["best_fitness"])
            ds_results["GridSearch"]["times"].append(gs_res["wall_clock_time"])

            # Manual
            mt_res = run_manual_tuning(X, y, dataset_name=ds_name, seed=seed)
            ds_results["Manual"]["f1_scores"].append(mt_res["best_fitness"])
            ds_results["Manual"]["times"].append(mt_res["wall_clock_time"])

        # Summary statistics
        for method in ds_results:
            ds_results[method]["stats"] = compute_summary_stats(
                ds_results[method]["f1_scores"]
            )

        # Statistical tests
        stat_tests = {}
        ga_scores = ds_results["GA"]["f1_scores"]
        for other in ["RandomSearch", "GridSearch", "Manual"]:
            try:
                stat_tests[f"GA_vs_{other}"] = wilcoxon_signed_rank(
                    ga_scores, ds_results[other]["f1_scores"]
                )
            except Exception as e:
                stat_tests[f"GA_vs_{other}"] = {"error": str(e)}
        ds_results["statistical_tests"] = stat_tests

        # Plots
        plot_comparison_boxplot(
            {m: {"f1_scores": ds_results[m]["f1_scores"]} for m in ds_results if m != "statistical_tests"},
            title=f"F1 Comparison — {ds_name}",
            save_path=CFG.PLOTS_DIR / f"comparison_boxplot_{ds_name}.png",
        )

        # Convergence comparison (use first seed's histories)
        if ga_histories and nsga_histories:
            plot_convergence_comparison(
                {"GA": ga_histories[0], "NSGA-II": nsga_histories[0]},
                title=f"Convergence — {ds_name}",
                save_path=CFG.PLOTS_DIR / f"convergence_{ds_name}.png",
            )

        # Confusion matrix for best GA pipeline
        if best_ga_individual is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                pipe = chromosome_to_pipeline(
                    best_ga_individual, X.shape[1], random_state=42
                )
                pipe.fit(X_train, y_train)
                plot_confusion_matrix(
                    pipe, X_test, y_test,
                    class_names=[str(c) for c in (class_names or range(len(set(y))))],
                    title=f"Best GA — {ds_name}",
                    save_path=CFG.PLOTS_DIR / f"confusion_matrix_{ds_name}.png",
                )
            except Exception as e:
                logger.warning(f"Could not plot confusion matrix for {ds_name}: {e}")

        plt.close("all")
        all_results[ds_name] = ds_results

    # Friedman test across datasets (if multiple)
    if len(datasets) > 1:
        methods = ["GA", "NSGA-II", "RandomSearch", "GridSearch", "Manual"]
        score_matrix = []
        for ds_name in datasets:
            row = []
            for m in methods:
                row.append(all_results[ds_name][m]["stats"]["mean"])
            score_matrix.append(row)

        try:
            all_results["_friedman"] = friedman_test(score_matrix, methods)
        except Exception as e:
            all_results["_friedman"] = {"error": str(e)}

    # Generate tables
    table_data = {}
    for ds_name in datasets:
        table_data[ds_name] = {}
        for m in ["GA", "NSGA-II", "RandomSearch", "GridSearch", "Manual"]:
            s = all_results[ds_name][m]["stats"]
            table_data[ds_name][m] = {
                "f1": s["mean"],
                "f1_mean": s["mean"],
                "f1_std": s["std"],
            }

    generate_markdown_table(
        table_data,
        output_path=CFG.TABLES_DIR / "comparison_table.md",
    )
    generate_latex_table(
        table_data,
        caption="Comparison of Pipeline Optimization Methods — F1 Score (macro)",
        output_path=CFG.TABLES_DIR / "comparison_table.tex",
    )

    total_time = time.perf_counter() - total_start
    all_results["_meta"] = {
        "total_time_seconds": total_time,
        "n_runs": len(seeds),
        "dry_run": dry_run,
    }

    output_path = CFG.RESULTS_DIR / "comparison_results.json"

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    output_path.write_text(json.dumps(all_results, indent=2, default=_serialize))
    logger.info(f"\nAll results saved to {output_path}")
    logger.info(f"Total experiment time: {total_time:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full comparison experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: minimal config on iris only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_experiment(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
