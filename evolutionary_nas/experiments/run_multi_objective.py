"""
Run Multi-Objective NAS (NSGA-II)
=================================
Searches for Pareto-optimal architectures (accuracy vs. parameters)
using NSGA-II on both MLP and CNN search spaces.

Usage:
    python -m experiments.run_multi_objective
    python -m experiments.run_multi_objective --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, set_seed
from training.datasets import load_dataset, get_dataset_info
from fitness.evaluator import FitnessEvaluator
from fitness.cache import FitnessCache
from evolution.multi_objective import run_nsga2
from search_space.genome_encoder import decode, describe
from visualization.pareto_front import plot_pareto_front_matplotlib
from visualization.fitness_curves import plot_fitness_evolution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Objective NAS (NSGA-II)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--net-type", type=str, default=None, choices=["mlp", "cnn"])
    args = parser.parse_args()

    if args.dry_run:
        datasets = ["MNIST"]
        net_types = ["mlp"]
        seeds = [42]
        pop_size, n_gen, fast_epochs = 10, 3, 2
    else:
        datasets = [args.dataset] if args.dataset else ["FashionMNIST", "CIFAR10"]
        net_types = [args.net_type] if args.net_type else ["mlp", "cnn"]
        seeds = CFG.RANDOM_SEEDS[:3]
        pop_size, n_gen = CFG.POP_SIZE_NSGA, CFG.N_GEN_NSGA
        fast_epochs = CFG.FAST_EPOCHS

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for dataset_name in datasets:
        info = get_dataset_info(dataset_name)

        for net_type in net_types:
            flatten = net_type == "mlp"
            key = f"{dataset_name}_{net_type}"
            mo_results = []

            for seed in seeds:
                logger.info(f"=== NSGA-II | {net_type.upper()} | {dataset_name} | seed={seed} ===")
                set_seed(seed)

                train_loader, test_loader = load_dataset(
                    dataset_name, batch_size=64, flatten=flatten
                )

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

                log_path = (
                    results_dir / "logs" / f"nsga2_{net_type}_{dataset_name}_s{seed}.json"
                )
                result = run_nsga2(
                    evaluator=evaluator,
                    net_type=net_type,
                    pop_size=pop_size,
                    n_gen=n_gen,
                    seed=seed,
                    log_path=log_path,
                )

                pareto = result["pareto_front"]
                pareto_info = []
                for ind in pareto:
                    genome = list(ind)
                    acc = ind.fitness.values[0]
                    neg_params = ind.fitness.values[1]
                    params = -neg_params
                    desc = describe(genome, net_type)
                    pareto_info.append({
                        "genome": genome,
                        "accuracy": acc,
                        "params": params,
                        "description": desc,
                    })

                mo_results.append({
                    "seed": seed,
                    "pareto_size": len(pareto),
                    "pareto": pareto_info,
                })

                accuracies = [p["accuracy"] for p in pareto_info]
                params_list = [p["params"] for p in pareto_info]

                plot_pareto_front_matplotlib(
                    accuracies, params_list,
                    save_path=results_dir / "plots" / f"pareto_{net_type}_{dataset_name}_s{seed}.png",
                    title=f"Pareto Front — {net_type.upper()} {dataset_name} (seed {seed})",
                )

                if result.get("history"):
                    plot_fitness_evolution(
                        result["history"],
                        save_path=results_dir / "plots" / f"fitness_nsga2_{net_type}_{dataset_name}_s{seed}.png",
                        title=f"NSGA-II — {net_type.upper()} {dataset_name} (seed {seed})",
                    )

            all_results[key] = mo_results

    out_path = results_dir / "multi_objective_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
