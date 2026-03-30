"""
Manual tuning baseline.

Provides hand-crafted, sensible default pipelines that a competent ML
practitioner would typically try first. Evaluates them with the same
fitness function for fair comparison.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

from ..fitness.cache import FitnessCache
from ..fitness.evaluator import FitnessEvaluator
from ..genome.chromosome import chromosome_description

logger = logging.getLogger(__name__)

# Manually designed pipelines as chromosomes
MANUAL_PIPELINES: List[Dict[str, Any]] = [
    {
        "name": "StandardScaler + RandomForest(100)",
        "chromosome": [0.33, 0.0, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
    },
    {
        "name": "RobustScaler + GradientBoosting(100, lr=0.1)",
        "chromosome": [1.0, 0.0, 0.5, 0.0, 0.33, 0.33, 0.18, 0.17, 1.0, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "StandardScaler + SVC(C=1, rbf)",
        "chromosome": [0.33, 0.0, 0.5, 0.0, 0.5, 0.33, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "MinMaxScaler + KNN(5, distance)",
        "chromosome": [0.67, 0.0, 0.5, 0.0, 0.67, 0.14, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "StandardScaler + LogisticRegression(C=1)",
        "chromosome": [0.33, 0.0, 0.5, 0.0, 0.83, 0.33, 0.0, 0.22, 0.0, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "name": "StandardScaler + SelectKBest + MLP(128, relu)",
        "chromosome": [0.33, 0.33, 0.5, 0.0, 1.0, 0.47, 0.0, 0.3, 0.33, 0.0, 0.4, 0.5, 0.5],
    },
]


def run_manual_tuning(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate manually-tuned pipelines and return the best.

    Returns:
        Dictionary with: best_individual, best_fitness, best_name,
        best_description, all_results, wall_clock_time.
    """
    cache = FitnessCache()
    evaluator = FitnessEvaluator(
        X, y,
        dataset_name=dataset_name,
        cache=cache,
        random_state=seed,
    )
    n_features = X.shape[1]

    best_chrom = None
    best_f1 = -1.0
    best_time = 0.0
    best_name = ""
    all_results = []

    start = time.perf_counter()

    for mp in MANUAL_PIPELINES:
        chrom = mp["chromosome"]
        f1, train_time = evaluator.evaluate(chrom)
        all_results.append({
            "name": mp["name"],
            "f1": f1,
            "training_time": train_time,
        })
        logger.info(f"Manual: {mp['name']} | F1: {f1:.4f} | Time: {train_time:.4f}s")

        if f1 > best_f1:
            best_f1 = f1
            best_chrom = list(chrom)
            best_time = train_time
            best_name = mp["name"]

    wall_clock = time.perf_counter() - start

    return {
        "best_individual": best_chrom,
        "best_fitness": best_f1,
        "best_training_time": best_time,
        "best_name": best_name,
        "best_description": chromosome_description(best_chrom, n_features),
        "all_results": all_results,
        "wall_clock_time": wall_clock,
        "n_features": n_features,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    results = run_manual_tuning(X, y, dataset_name="iris", seed=42)
    print(f"\nBest: {results['best_name']} | F1: {results['best_fitness']:.4f}")
