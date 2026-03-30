"""
Cross-Validation
================
Multi-seed final evaluation of best architectures with full training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from config import CFG, set_seed
from search_space.genome_encoder import decode
from models.mlp_builder import build_mlp
from models.cnn_builder import build_cnn
from models.model_utils import count_parameters
from training.trainer import train_model
from training.datasets import load_dataset, get_dataset_info
from fitness.metrics import compute_accuracy, compute_f1

logger = logging.getLogger(__name__)


def multi_seed_evaluation(
    genome: List[float],
    net_type: str,
    dataset_name: str,
    seeds: List[int] | None = None,
    epochs: int = 30,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate a single architecture across multiple seeds.

    Returns per-seed and aggregate metrics.
    """
    if seeds is None:
        seeds = CFG.RANDOM_SEEDS

    config = decode(genome, net_type)
    info = get_dataset_info(dataset_name)
    flatten = net_type == "mlp"
    batch_size = config.get("batch_size", 64)

    per_seed_results: List[Dict[str, float]] = []

    for seed in seeds:
        set_seed(seed)
        train_loader, test_loader = load_dataset(
            dataset_name, batch_size=batch_size, flatten=flatten
        )

        if net_type == "mlp":
            model = build_mlp(config, info["input_dim_flat"], info["num_classes"])
        else:
            model = build_cnn(config, info["in_channels"], info["num_classes"])

        params = count_parameters(model)

        try:
            result = train_model(
                model, train_loader, test_loader,
                epochs=epochs,
                optimizer_name=config.get("optimizer", "adam"),
                lr=config.get("learning_rate", 1e-3),
                weight_decay=config.get("weight_decay", 1e-5),
                device=device,
                patience=CFG.EARLY_STOP_PATIENCE,
            )
            acc = compute_accuracy(model, test_loader, device)
            f1 = compute_f1(model, test_loader, device)
        except Exception as e:
            logger.warning(f"Evaluation failed for seed {seed}: {e}")
            acc, f1 = 0.0, 0.0
            result = type("R", (), {"best_val_acc": 0.0, "total_epochs": 0,
                                     "elapsed_seconds": 0.0,
                                     "train_losses": [], "val_losses": [],
                                     "train_accs": [], "val_accs": []})()

        per_seed_results.append({
            "seed": seed,
            "test_accuracy": acc,
            "test_f1": f1,
            "param_count": params,
            "best_val_acc": result.best_val_acc,
            "epochs_trained": result.total_epochs,
            "elapsed_seconds": result.elapsed_seconds,
        })

    accs = [r["test_accuracy"] for r in per_seed_results]
    f1s = [r["test_f1"] for r in per_seed_results]

    return {
        "per_seed": per_seed_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "param_count": per_seed_results[0]["param_count"],
    }


if __name__ == "__main__":
    print("Cross-validation module ready.")
