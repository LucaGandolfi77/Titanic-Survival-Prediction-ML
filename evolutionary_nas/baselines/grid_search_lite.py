"""
Grid Search Lite
================
Small grid search over a reduced subspace of architectures.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config import CFG, set_seed
from search_space.genome_encoder import decode, describe, encode, repair
from fitness.evaluator import FitnessEvaluator

logger = logging.getLogger(__name__)


def run_grid_search_mlp(
    evaluator: FitnessEvaluator,
    seed: int = 42,
) -> Dict[str, Any]:
    """Grid search over a reduced MLP subspace."""
    set_seed(seed)

    grid = {
        "n_layers": [1, 2, 3],
        "hidden_sizes": [[64], [128, 64], [256, 128, 64]],
        "activation": ["relu", "gelu"],
        "dropout_rate": [0.0, 0.3],
        "use_batch_norm": [False, True],
        "optimizer": ["adam"],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4],
        "batch_size": [64],
    }

    configs = []
    for hs in grid["hidden_sizes"]:
        for act in grid["activation"]:
            for do in grid["dropout_rate"]:
                for bn in grid["use_batch_norm"]:
                    for lr in grid["learning_rate"]:
                        configs.append({
                            "n_layers": len(hs),
                            "hidden_sizes": hs,
                            "activation": act,
                            "dropout_rate": do,
                            "use_batch_norm": bn,
                            "optimizer": "adam",
                            "learning_rate": lr,
                            "weight_decay": 1e-4,
                            "batch_size": 64,
                        })

    return _evaluate_configs(evaluator, configs, "mlp")


def run_grid_search_cnn(
    evaluator: FitnessEvaluator,
    seed: int = 42,
) -> Dict[str, Any]:
    """Grid search over a reduced CNN subspace."""
    set_seed(seed)

    filter_sets = [[16, 32], [32, 64], [32, 64, 128]]
    configs = []
    for fs in filter_sets:
        for act in ["relu", "gelu"]:
            for bn in [False, True]:
                for lr in [1e-3, 5e-4]:
                    configs.append({
                        "n_conv_blocks": len(fs),
                        "filters": fs,
                        "kernel_size": 3,
                        "use_depthwise": False,
                        "use_skip_conn": True,
                        "pooling_type": "max",
                        "activation": act,
                        "dropout_rate": 0.2,
                        "use_batch_norm": bn,
                        "dense_layers": 1,
                        "dense_width": 128,
                        "optimizer": "adam",
                        "learning_rate": lr,
                        "weight_decay": 1e-4,
                        "batch_size": 64,
                    })

    return _evaluate_configs(evaluator, configs, "cnn")


def _evaluate_configs(
    evaluator: FitnessEvaluator,
    configs: List[Dict],
    net_type: str,
) -> Dict[str, Any]:
    best_acc = 0.0
    best_genome: Optional[List[float]] = None
    all_results: List[Dict] = []
    start = time.perf_counter()

    for cfg in configs:
        try:
            genome = encode(cfg, net_type)
            genome = repair(genome, net_type)
            acc, params = evaluator.evaluate(genome)
        except Exception as e:
            logger.warning(f"Grid search config failed: {e}")
            acc, params = 0.0, float(CFG.MAX_PARAMS)
            genome = [0.0] * 14

        all_results.append({"accuracy": acc, "param_count": params})
        if acc > best_acc:
            best_acc = acc
            best_genome = genome

    elapsed = time.perf_counter() - start
    accs = [r["accuracy"] for r in all_results]
    return {
        "best_genome": best_genome,
        "best_accuracy": best_acc,
        "best_description": describe(best_genome, net_type) if best_genome else "",
        "all_accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "n_evaluated": len(configs),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    print("Grid search module ready.")
