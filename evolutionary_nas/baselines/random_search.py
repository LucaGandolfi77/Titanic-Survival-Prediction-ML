"""
Random Search Baseline
======================
Randomly sample architectures from the same search space used by the GA.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config import CFG, set_seed
from search_space.mlp_space import random_mlp_genome
from search_space.cnn_space import random_cnn_genome
from search_space.genome_encoder import decode, describe, repair
from fitness.evaluator import FitnessEvaluator

logger = logging.getLogger(__name__)


def run_random_search(
    evaluator: FitnessEvaluator,
    net_type: str,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate n_samples random architectures and return bestresult."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    gen_fn = random_mlp_genome if net_type == "mlp" else random_cnn_genome

    best_acc = 0.0
    best_genome: Optional[List[float]] = None
    all_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for i in range(n_samples):
        genome = gen_fn(rng)
        genome = repair(genome, net_type)
        acc, params = evaluator.evaluate(genome)
        all_results.append({
            "genome": genome,
            "accuracy": acc,
            "param_count": params,
        })
        if acc > best_acc:
            best_acc = acc
            best_genome = genome

    elapsed = time.perf_counter() - start_time
    accs = [r["accuracy"] for r in all_results]

    return {
        "best_genome": best_genome,
        "best_accuracy": best_acc,
        "best_description": describe(best_genome, net_type) if best_genome else "",
        "all_accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "n_evaluated": n_samples,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    print("Random search module ready.")
