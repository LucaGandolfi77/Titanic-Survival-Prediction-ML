"""
Evolutionary NAS — Global Configuration
========================================
All hyperparameters, paths, and constants for the evolutionary neural
architecture search system. Imported as a singleton by every module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch


def _detect_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class NASConfig:
    # ── Reproducibility ──────────────────────────────────────────────
    RANDOM_SEEDS: List[int] = field(
        default_factory=lambda: [42, 7, 13, 99, 100, 21, 55, 77, 11, 33]
    )
    N_RUNS: int = 10

    # ── GA (single-objective) ────────────────────────────────────────
    POP_SIZE_GA: int = 50
    N_GEN_GA: int = 40
    TOURNAMENT_SIZE: int = 3
    CXPB: float = 0.7
    MUTPB: float = 0.15
    HOF_SIZE: int = 5
    LAMBDA_PARAM_PENALTY: float = 0.5

    # ── NSGA-II (multi-objective) ────────────────────────────────────
    POP_SIZE_NSGA: int = 100
    N_GEN_NSGA: int = 50

    # ── Architecture constraints ─────────────────────────────────────
    MAX_PARAMS: int = 500_000
    MLP_MAX_LAYERS: int = 6
    CNN_MAX_BLOCKS: int = 5

    # ── Training ─────────────────────────────────────────────────────
    FAST_EPOCHS: int = 5
    FULL_EPOCHS: int = 30
    MAX_EVAL_SECONDS: int = 120
    EARLY_STOP_PATIENCE: int = 7
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 3

    # ── Surrogate ────────────────────────────────────────────────────
    SURROGATE_WARMUP: int = 50
    SURROGATE_TOPK: float = 0.3
    SURROGATE_RETRAIN_EVERY: int = 5

    # ── Predictive early stopping ────────────────────────────────────
    PREDICTOR_LOOKBACK: int = 3
    PREDICTOR_THRESHOLD: float = 0.0

    # ── Mutation parameters ──────────────────────────────────────────
    SBX_ETA: float = 15.0
    GAUSSIAN_SIGMA: float = 0.1

    # ── Paths ────────────────────────────────────────────────────────
    RESULTS_DIR: Path = Path("experiments/results")
    PLOTS_DIR: Path = Path("experiments/results/plots")
    LOGS_DIR: Path = Path("experiments/results/logs")
    CHECKPOINTS_DIR: Path = Path("experiments/results/checkpoints")
    TABLES_DIR: Path = Path("experiments/results/tables")

    # ── Device ───────────────────────────────────────────────────────
    DEVICE: str = field(default_factory=_detect_device)

    # ── Datasets ─────────────────────────────────────────────────────
    DATASETS: List[str] = field(
        default_factory=lambda: ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
    )

    def ensure_dirs(self) -> None:
        for d in [self.RESULTS_DIR, self.PLOTS_DIR, self.LOGS_DIR,
                  self.CHECKPOINTS_DIR, self.TABLES_DIR]:
            d.mkdir(parents=True, exist_ok=True)


CFG = NASConfig()


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


if __name__ == "__main__":
    print(f"Device: {CFG.DEVICE}")
    print(f"Seeds:  {CFG.RANDOM_SEEDS}")
    CFG.ensure_dirs()
    print("Directories created.")
