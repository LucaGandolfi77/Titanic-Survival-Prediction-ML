"""
Random search baseline.

Samples random chromosomes from the same search space used by the GA
and evaluates them with the same fitness function, ensuring a fair
comparison. Equivalent to RandomizedSearchCV over the pipeline space.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

from ..fitness.cache import FitnessCache
from ..fitness.evaluator import FitnessEvaluator
from ..genome.chromosome import CHROMOSOME_LENGTH, chromosome_description, random_chromosome

logger = logging.getLogger(__name__)


def run_random_search(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    n_iter: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate n_iter random chromosomes and return the best.

    Uses the same fitness evaluator and search space as the evolutionary
    approach to ensure a fair comparison.

    Returns:
        Dictionary with: best_individual, best_fitness, best_description,
        all_results, wall_clock_time, eval_count.
    """
    rng = np.random.default_rng(seed)
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

    for i in range(n_iter):
        chrom = random_chromosome(rng)
        f1, train_time = evaluator.evaluate(chrom)
        all_results.append({"f1": f1, "training_time": train_time})

        if f1 > best_f1:
            best_f1 = f1
            best_chrom = chrom
            best_time = train_time

        if (i + 1) % 50 == 0:
            logger.info(
                f"RandomSearch iter {i+1}/{n_iter} | Best F1: {best_f1:.4f}"
            )

    wall_clock = time.perf_counter() - start

    return {
        "best_individual": best_chrom,
        "best_fitness": best_f1,
        "best_training_time": best_time,
        "best_description": chromosome_description(best_chrom, n_features),
        "all_results": all_results,
        "wall_clock_time": wall_clock,
        "eval_count": evaluator.eval_count,
        "n_features": n_features,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    results = run_random_search(X, y, dataset_name="iris", n_iter=50, seed=42)
    print(f"Best F1: {results['best_fitness']:.4f}")
    print(f"Pipeline: {results['best_description']}")
    print(f"Wall clock: {results['wall_clock_time']:.1f}s")
