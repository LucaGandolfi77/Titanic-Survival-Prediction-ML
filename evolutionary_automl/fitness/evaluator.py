"""
Fitness evaluator with cross-validation, timing, caching, and timeout.

This is the central evaluation engine used by both single-objective and
multi-objective evolutionary strategies. It wraps the full pipeline build →
cross-validate → measure cycle with exception handling and configurable
time budgets.
"""
from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..config import CFG
from ..genome.chromosome import chromosome_to_pipeline
from ..search_space.validators import repair_chromosome
from .cache import FitnessCache
from .metrics import compute_f1_cv, count_features_used, measure_training_time

logger = logging.getLogger(__name__)

# Suppress sklearn convergence warnings during mass evaluation
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class FitnessEvaluator:
    """Evaluates the fitness of a chromosome on a given dataset.

    Primary objective: F1 macro (maximize)
    Secondary objective: training time in seconds (minimize)

    Supports caching and per-evaluation timeout.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "unknown",
        cv_folds: int = CFG.CV_FOLDS,
        max_eval_seconds: int = CFG.MAX_EVAL_SECONDS,
        cache: Optional[FitnessCache] = None,
        random_state: int = 42,
    ) -> None:
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.n_features = X.shape[1]
        self.cv_folds = cv_folds
        self.max_eval_seconds = max_eval_seconds
        self.cache = cache if cache is not None else FitnessCache()
        self.random_state = random_state
        self._eval_count = 0

    def evaluate(self, chromosome: List[float]) -> Tuple[float, float]:
        """Evaluate a single chromosome, returning (f1, training_time).

        Returns (0.0, max_eval_seconds) on failure or timeout.
        """
        chromosome = repair_chromosome(list(chromosome))

        cached = self.cache.get(chromosome, self.dataset_name)
        if cached is not None:
            return cached

        self._eval_count += 1

        try:
            result = self._evaluate_with_timeout(chromosome)
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            result = (0.0, float(self.max_eval_seconds))

        self.cache.put(chromosome, self.dataset_name, result)
        return result

    def _evaluate_with_timeout(
        self, chromosome: List[float]
    ) -> Tuple[float, float]:
        """Run evaluation with a timeout guard."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._evaluate_inner, chromosome)
            try:
                return future.result(timeout=self.max_eval_seconds)
            except TimeoutError:
                logger.warning("Evaluation timed out")
                return (0.0, float(self.max_eval_seconds))
            except Exception as e:
                logger.debug(f"Inner evaluation error: {e}")
                return (0.0, float(self.max_eval_seconds))

    def _evaluate_inner(self, chromosome: List[float]) -> Tuple[float, float]:
        """Core evaluation: build pipeline, cross-validate, measure time."""
        pipeline = chromosome_to_pipeline(
            chromosome, self.n_features, self.random_state
        )

        start = time.perf_counter()
        f1 = compute_f1_cv(
            pipeline, self.X, self.y,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
        )
        elapsed = time.perf_counter() - start

        return (float(f1), float(elapsed))

    def evaluate_single_objective(self, chromosome: List[float]) -> Tuple[float]:
        """DEAP-compatible single-objective fitness (maximize F1)."""
        f1, _ = self.evaluate(chromosome)
        return (f1,)

    def evaluate_multi_objective(self, chromosome: List[float]) -> Tuple[float, float]:
        """DEAP-compatible multi-objective fitness (maximize F1, minimize time)."""
        f1, train_time = self.evaluate(chromosome)
        return (f1, train_time)

    @property
    def eval_count(self) -> int:
        return self._eval_count


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    evaluator = FitnessEvaluator(X, y, dataset_name="iris", cv_folds=3)
    chromosome = [0.33, 0.0, 0.5, 0.0, 0.17, 0.5, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    result = evaluator.evaluate(chromosome)
    print(f"F1={result[0]:.4f}, Time={result[1]:.4f}s")
    print(f"Cache: {evaluator.cache}")
