"""
Grid search baseline on a reduced subspace.

Evaluates a fixed grid of pipeline configurations covering the most
common combinations of scaler, feature selector, and classifier with
a handful of hyperparameter values.
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

# Pre-defined chromosome grid (reduced subspace)
# Each chromosome encodes: [scaler, feat_sel, k_ratio, dim_red, clf, hp0..hp7]
_GRID: List[List[float]] = []

# Scaler options: 0.0 (none), 0.33 (standard), 0.67 (minmax), 1.0 (robust)
_SCALERS = [0.0, 0.33, 1.0]
# Feature sel: 0.0 (none), 0.33 (selectkbest)
_FEAT_SELS = [0.0, 0.33]
# Classifiers with default hyperparameters
_CLF_CONFIGS = [
    # RandomForest defaults
    [0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5],
    # GradientBoosting defaults
    [0.33, 0.33, 0.3, 0.17, 0.5, 0.5, 0.5, 0.5],
    # KNN defaults
    [0.67, 0.14, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5],
    # LogisticRegression defaults
    [0.83, 0.33, 0.33, 0.22, 0.0, 0.5, 0.5, 0.5],
    # SVC defaults
    [0.5, 0.33, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
]

for sc in _SCALERS:
    for fs in _FEAT_SELS:
        for clf_hp in _CLF_CONFIGS:
            chrom = [sc, fs, 0.5, 0.0] + clf_hp[:1] + clf_hp[1:]
            if len(chrom) < 13:
                chrom.extend([0.5] * (13 - len(chrom)))
            _GRID.append(chrom[:13])


def run_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate the pre-defined grid and return the best configuration.

    Returns:
        Dictionary with: best_individual, best_fitness, best_description,
        grid_size, wall_clock_time, eval_count.
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
    all_results = []

    start = time.perf_counter()

    for i, chrom in enumerate(_GRID):
        f1, train_time = evaluator.evaluate(chrom)
        all_results.append({"f1": f1, "training_time": train_time})

        if f1 > best_f1:
            best_f1 = f1
            best_chrom = list(chrom)
            best_time = train_time

        if (i + 1) % 10 == 0:
            logger.info(
                f"GridSearch config {i+1}/{len(_GRID)} | Best F1: {best_f1:.4f}"
            )

    wall_clock = time.perf_counter() - start

    return {
        "best_individual": best_chrom,
        "best_fitness": best_f1,
        "best_training_time": best_time,
        "best_description": chromosome_description(best_chrom, n_features),
        "all_results": all_results,
        "grid_size": len(_GRID),
        "wall_clock_time": wall_clock,
        "eval_count": evaluator.eval_count,
        "n_features": n_features,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    results = run_grid_search(X, y, dataset_name="iris", seed=42)
    print(f"Grid size: {results['grid_size']}")
    print(f"Best F1: {results['best_fitness']:.4f}")
    print(f"Pipeline: {results['best_description']}")
