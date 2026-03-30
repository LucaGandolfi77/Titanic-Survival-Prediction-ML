"""
Run MLP NAS Experiment
======================
Full MLP architecture search across datasets and seeds.

Usage:
    python -m experiments.run_mlp_search
    python -m experiments.run_mlp_search --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, NASConfig, set_seed
from training.datasets import load_dataset, get_dataset_info
from fitness.evaluator import FitnessEvaluator
from fitness.cache import FitnessCache
from evolution.single_objective import run_single_objective_ga
from search_space.genome_encoder import decode
from models.mlp_builder import build_mlp
from models.model_utils import count_parameters
from training.trainer import train_model
from visualization.fitness_curves import plot_fitness_evolution
from visualization.comparison_boxplot import plot_diversity_curve
from visualization.architecture_diagram import plot_architecture_mlp
from visualization.weight_distribution import plot_weight_distribution, plot_learning_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP NAS Experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick run: 3 gens, pop=10, MNIST only")
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.dry_run:
        datasets = ["MNIST"]
        seeds = [42]
        pop_size, n_gen, fast_epochs = 10, 3, 2
        use_surrogate = False
    else:
        datasets = [args.dataset] if args.dataset else ["MNIST", "FashionMNIST"]
        seeds = [args.seed] if args.seed else CFG.RANDOM_SEEDS
        pop_size, n_gen = CFG.POP_SIZE_GA, CFG.N_GEN_GA
        fast_epochs = CFG.FAST_EPOCHS
        use_surrogate = not args.no_surrogate

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name in datasets:
        info = get_dataset_info(dataset_name)
        dataset_results = []

        for seed in seeds:
            logger.info(f"=== MLP Search | {dataset_name} | seed={seed} ===")
            set_seed(seed)

            train_loader, test_loader = load_dataset(
                dataset_name, batch_size=64, flatten=True
            )

            cache = FitnessCache()
            evaluator = FitnessEvaluator(
                train_loader=train_loader,
                val_loader=test_loader,
                dataset_name=dataset_name,
                net_type="mlp",
                input_dim=info["input_dim_flat"],
                num_classes=info["num_classes"],
                device=CFG.DEVICE,
                fast_epochs=fast_epochs,
                cache=cache,
            )

            log_path = results_dir / "logs" / f"mlp_{dataset_name}_seed{seed}.json"
            result = run_single_objective_ga(
                evaluator=evaluator,
                net_type="mlp",
                pop_size=pop_size,
                n_gen=n_gen,
                seed=seed,
                use_surrogate=use_surrogate,
                log_path=log_path,
            )

            dataset_results.append({
                "seed": seed,
                "best_fitness": result["best_fitness"],
                "best_description": result["best_description"],
                "best_genome": result["best_genome"],
                "cache_stats": result["cache_stats"],
            })

            # Plot fitness evolution
            plot_fitness_evolution(
                result["history"],
                save_path=results_dir / "plots" / f"fitness_mlp_{dataset_name}_s{seed}.png",
                title=f"MLP NAS — {dataset_name} (seed {seed})",
            )

            # Plot diversity
            plot_diversity_curve(
                result["history"],
                save_path=results_dir / "plots" / f"diversity_mlp_{dataset_name}_s{seed}.png",
            )

        all_results[dataset_name] = dataset_results

        # Full train the best from this dataset
        best_run = max(dataset_results, key=lambda r: r["best_fitness"])
        best_genome = best_run["best_genome"]
        best_config = decode(best_genome, "mlp")

        logger.info(f"Full training best MLP for {dataset_name}: {best_run['best_description']}")
        set_seed(42)
        train_loader, test_loader = load_dataset(
            dataset_name, batch_size=best_config["batch_size"], flatten=True
        )
        model = build_mlp(best_config, info["input_dim_flat"], info["num_classes"])
        logger.info(f"Params: {count_parameters(model):,}")

        train_result = train_model(
            model, train_loader, test_loader,
            epochs=CFG.FULL_EPOCHS if not args.dry_run else 3,
            optimizer_name=best_config["optimizer"],
            lr=best_config["learning_rate"],
            weight_decay=best_config["weight_decay"],
            device=CFG.DEVICE,
            patience=CFG.EARLY_STOP_PATIENCE,
        )

        # Architecture diagram
        plot_architecture_mlp(
            best_config,
            save_path=results_dir / "plots" / f"arch_mlp_{dataset_name}.png",
            title=f"Best MLP — {dataset_name} (acc={train_result.best_val_acc:.4f})",
        )

        # Weight distribution
        plot_weight_distribution(
            model,
            save_path=results_dir / "plots" / f"weights_mlp_{dataset_name}.png",
        )

        # Learning curve
        plot_learning_curve(
            train_result.train_losses, train_result.val_losses,
            train_result.train_accs, train_result.val_accs,
            save_path=results_dir / "plots" / f"learning_curve_mlp_{dataset_name}.png",
        )

    # Save all results
    out_path = results_dir / "mlp_search_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
