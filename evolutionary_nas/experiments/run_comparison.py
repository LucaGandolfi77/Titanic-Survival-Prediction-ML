"""
Run Comparison: NAS vs Baselines
================================
Compares evolutionary NAS against random search, grid search,
and fixed architectures with statistical testing.

Usage:
    python -m experiments.run_comparison
    python -m experiments.run_comparison --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, set_seed
from training.datasets import load_dataset, get_dataset_info
from fitness.evaluator import FitnessEvaluator
from fitness.cache import FitnessCache
from evolution.single_objective import run_single_objective_ga
from baselines.random_search import run_random_search
from baselines.grid_search_lite import run_grid_search_mlp, run_grid_search_cnn
from baselines.fixed_small import get_fixed_mlp_configs, get_fixed_cnn_configs
from search_space.genome_encoder import decode
from models.mlp_builder import build_mlp
from models.cnn_builder import build_cnn
from models.model_utils import count_parameters
from training.trainer import train_model
from evaluation.statistical_tests import (
    wilcoxon_signed_rank, friedman_test, compute_summary_stats,
)
from evaluation.report_generator import generate_markdown_table, generate_latex_table, save_tables
from visualization.comparison_boxplot import plot_comparison_boxplot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_fixed(
    configs: list[dict],
    net_type: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
) -> list[float]:
    """Evaluate a list of fixed configs and return accuracies."""
    accs = []
    for cfg in configs:
        if net_type == "mlp":
            model = build_mlp(cfg, info["input_dim"], info["num_classes"])
        else:
            model = build_cnn(cfg, info["in_channels"], info["num_classes"])

        result = train_model(
            model, train_loader, test_loader,
            epochs=epochs,
            optimizer_name=cfg.get("optimizer", "adam"),
            lr=cfg.get("learning_rate", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            device=CFG.DEVICE,
        )
        accs.append(result.test_acc)
    return accs


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS vs Baselines Comparison")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--net-type", type=str, default=None, choices=["mlp", "cnn"])
    args = parser.parse_args()

    if args.dry_run:
        datasets = ["MNIST"]
        net_types = ["mlp"]
        seeds = [42, 7]
        pop_size, n_gen = 10, 3
        fast_epochs, full_epochs = 2, 3
        n_random = 20
    else:
        datasets = [args.dataset] if args.dataset else ["FashionMNIST", "CIFAR10"]
        net_types = [args.net_type] if args.net_type else ["mlp", "cnn"]
        seeds = CFG.RANDOM_SEEDS[:5]
        pop_size, n_gen = CFG.POP_SIZE_GA, CFG.N_GEN_GA
        fast_epochs, full_epochs = CFG.FAST_EPOCHS, CFG.FULL_EPOCHS
        n_random = pop_size * n_gen

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_all = {}

    for dataset_name in datasets:
        info = get_dataset_info(dataset_name)

        for net_type in net_types:
            flatten = net_type == "mlp"
            key = f"{dataset_name}_{net_type}"
            logger.info(f"=== Comparison | {net_type.upper()} | {dataset_name} ===")

            method_results: dict[str, list[float]] = {
                "NAS-GA": [],
                "RandomSearch": [],
                "GridSearch": [],
                "FixedSmall": [],
            }

            for seed in seeds:
                set_seed(seed)
                train_loader, test_loader = load_dataset(
                    dataset_name, batch_size=64, flatten=flatten
                )

                # --- NAS-GA ---
                cache = FitnessCache()
                evaluator = FitnessEvaluator(
                    train_loader=train_loader,
                    val_loader=test_loader,
                    dataset_name=dataset_name,
                    net_type=net_type,
                    in_channels=info["in_channels"],
                    num_classes=info["num_classes"],
                    device=CFG.DEVICE,
                    fast_epochs=fast_epochs,
                    cache=cache,
                )
                ga_result = run_single_objective_ga(
                    evaluator=evaluator,
                    net_type=net_type,
                    pop_size=pop_size,
                    n_gen=n_gen,
                    seed=seed,
                    use_surrogate=False,
                )
                method_results["NAS-GA"].append(ga_result["best_fitness"])

                # --- Random Search ---
                random_res = run_random_search(
                    evaluator=evaluator,
                    net_type=net_type,
                    n_samples=n_random,
                    seed=seed,
                )
                method_results["RandomSearch"].append(random_res["best_fitness"])

                # --- Grid Search ---
                if net_type == "mlp":
                    grid_res = run_grid_search_mlp(evaluator=evaluator)
                else:
                    grid_res = run_grid_search_cnn(evaluator=evaluator)
                method_results["GridSearch"].append(grid_res["best_fitness"])

                # --- Fixed Small ---
                if net_type == "mlp":
                    fixed_configs = get_fixed_mlp_configs()
                else:
                    fixed_configs = get_fixed_cnn_configs()
                fixed_accs = evaluate_fixed(
                    fixed_configs, net_type, train_loader, test_loader,
                    info, epochs=fast_epochs,
                )
                method_results["FixedSmall"].append(max(fixed_accs))

            # Statistical analysis
            stats = {}
            for method, values in method_results.items():
                stats[method] = compute_summary_stats(values)

            nas_vals = method_results["NAS-GA"]
            stat_tests = {}
            for baseline in ["RandomSearch", "GridSearch", "FixedSmall"]:
                bvals = method_results[baseline]
                if len(nas_vals) >= 5:
                    wtest = wilcoxon_signed_rank(nas_vals, bvals)
                    stat_tests[f"NAS_vs_{baseline}"] = {
                        "statistic": wtest["statistic"],
                        "p_value": wtest["p_value"],
                        "significant": wtest["significant"],
                    }

            if len(method_results) >= 3:
                try:
                    groups = list(method_results.values())
                    ftest = friedman_test(groups)
                    stat_tests["friedman"] = ftest
                except Exception:
                    pass

            comparison_all[key] = {
                "results": {m: v for m, v in method_results.items()},
                "stats": stats,
                "tests": stat_tests,
            }

            # Boxplot
            plot_comparison_boxplot(
                method_results,
                save_path=results_dir / "plots" / f"comparison_{net_type}_{dataset_name}.png",
                title=f"NAS vs Baselines — {net_type.upper()} {dataset_name}",
            )

            # Tables
            rows = []
            for method, s in stats.items():
                rows.append({
                    "Method": method,
                    "Mean Acc": f"{s['mean']:.4f}",
                    "Std": f"{s['std']:.4f}",
                    "Min": f"{s['min']:.4f}",
                    "Max": f"{s['max']:.4f}",
                })

            md_table = generate_markdown_table(
                rows, columns=["Method", "Mean Acc", "Std", "Min", "Max"]
            )
            latex_table = generate_latex_table(
                rows, columns=["Method", "Mean Acc", "Std", "Min", "Max"],
                best_col="Mean Acc",
            )
            save_tables(
                md_table, latex_table,
                results_dir / "tables" / f"comparison_{net_type}_{dataset_name}",
            )

    out_path = results_dir / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(comparison_all, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
