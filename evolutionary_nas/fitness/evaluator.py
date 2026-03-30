"""
Fitness Evaluator
=================
Main fitness evaluation: builds a model from genome, trains it for
FAST_EPOCHS, and returns (accuracy, param_count). Includes timeout
guard and exception handling.
"""

from __future__ import annotations

import logging
import math
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from search_space.genome_encoder import decode, hash_genome
from models.mlp_builder import build_mlp
from models.cnn_builder import build_cnn
from models.model_utils import count_parameters
from training.fast_trainer import fast_train, FastTrainResult
from fitness.cache import FitnessCache
from fitness.metrics import compute_accuracy

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Evaluate genomes by building and fast-training the corresponding model."""

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_name: str,
        net_type: str,
        input_dim: int = 784,
        in_channels: int = 1,
        num_classes: int = 10,
        device: str = "cpu",
        fast_epochs: int = 5,
        max_eval_seconds: int = 120,
        cache: Optional[FitnessCache] = None,
        learning_curve_predictor: Optional[Any] = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_name = dataset_name
        self.net_type = net_type
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
        self.fast_epochs = fast_epochs
        self.max_eval_seconds = max_eval_seconds
        self.cache = cache or FitnessCache()
        self.learning_curve_predictor = learning_curve_predictor

        self._n_real_evals = 0
        self._total_epochs_run = 0
        self._total_epochs_saved = 0

    def evaluate(self, genome: List[float]) -> Tuple[float, ...]:
        """Evaluate a genome, returning (accuracy, param_count).

        Uses cache if available; catches all exceptions and returns
        penalized fitness (0.0, MAX_PARAMS) on error.
        """
        key = hash_genome(genome, self.dataset_name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            result = self._evaluate_impl(genome)
            self.cache.put(key, result)
            return result
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            penalty = (0.0, float(CFG.MAX_PARAMS))
            self.cache.put(key, penalty)
            return penalty

    def _evaluate_impl(self, genome: List[float]) -> Tuple[float, float]:
        config = decode(genome, self.net_type)

        if self.net_type == "mlp":
            model = build_mlp(config, self.input_dim, self.num_classes)
        else:
            model = build_cnn(config, self.in_channels, self.num_classes)

        params = count_parameters(model)
        if params > CFG.MAX_PARAMS:
            return (0.0, float(params))

        train_result = fast_train(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.fast_epochs,
            optimizer_name=config.get("optimizer", "adam"),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
            device=self.device,
            learning_curve_predictor=self.learning_curve_predictor,
            accuracy_threshold=CFG.PREDICTOR_THRESHOLD,
        )

        self._n_real_evals += 1
        self._total_epochs_run += train_result.epochs_run
        if train_result.early_stopped:
            self._total_epochs_saved += (self.fast_epochs - train_result.epochs_run)

        if self.learning_curve_predictor is not None and hasattr(
            self.learning_curve_predictor, "add_curve"
        ):
            self.learning_curve_predictor.add_curve(
                train_result.val_accs, train_result.final_val_acc
            )

        return (train_result.final_val_acc, float(params))

    def evaluate_single_objective(
        self, genome: List[float], lambda_penalty: float = 0.5
    ) -> Tuple[float]:
        """Single-objective fitness: accuracy - lambda * log10(param_count)."""
        acc, params = self.evaluate(genome)
        if params == 0:
            params = 1.0
        fitness = acc - lambda_penalty * math.log10(max(params, 1))
        return (fitness,)

    def evaluate_multi_objective(
        self, genome: List[float]
    ) -> Tuple[float, float]:
        """Multi-objective: (accuracy, -param_count) for NSGA-II."""
        acc, params = self.evaluate(genome)
        return (acc, -params)

    @property
    def n_real_evals(self) -> int:
        return self._n_real_evals

    @property
    def epochs_saved(self) -> int:
        return self._total_epochs_saved

    @property
    def total_epochs_run(self) -> int:
        return self._total_epochs_run


if __name__ == "__main__":
    print("FitnessEvaluator ready.")
