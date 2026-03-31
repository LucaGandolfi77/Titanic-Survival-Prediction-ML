"""
Configuration — Ensemble Learning Study
========================================
Central repository for all experimental constants, paths, and
hyperparameter grids used across the project.  Frozen dataclass
ensures values cannot be accidentally mutated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class StudyConfig:
    # ── Seeds & repetitions ───────────────────────────────────────
    RANDOM_SEEDS: tuple[int, ...] = (42, 7, 13, 99, 100, 21, 55, 77, 11, 33, 3, 17, 88, 44, 66)
    N_RUNS: int = 15
    CV_FOLDS: int = 5
    CV_REPEATS: int = 3

    # ── Data perturbation grids ───────────────────────────────────
    DATASET_SIZES: list[int] = field(default_factory=lambda: [30, 50, 100, 200, 500, 1000, 2000])
    LABEL_NOISE_RATES: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.30])
    OUTLIER_FRACTIONS: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.10, 0.20])
    IMBALANCE_RATIOS: list[str] = field(default_factory=lambda: ["1:1", "1:2", "1:5", "1:10", "1:20"])

    # ── Ensemble hyperparameters ──────────────────────────────────
    N_ESTIMATORS_SWEEP: list[int] = field(default_factory=lambda: [5, 10, 25, 50, 100, 200])

    BAGGING_N_ESTIMATORS: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    RF_N_ESTIMATORS: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    RF_MAX_FEATURES: list = field(default_factory=lambda: ["sqrt", "log2", None])
    RF_MAX_DEPTH: list = field(default_factory=lambda: [None, 5, 10])
    ADA_N_ESTIMATORS: list[int] = field(default_factory=lambda: [50, 100, 200])
    ADA_LEARNING_RATE: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    GB_N_ESTIMATORS: list[int] = field(default_factory=lambda: [50, 100, 200])
    GB_LEARNING_RATE: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    GB_MAX_DEPTH: list[int] = field(default_factory=lambda: [3, 5])
    GB_SUBSAMPLE: list[float] = field(default_factory=lambda: [0.8, 1.0])

    # ── Statistical testing ───────────────────────────────────────
    SIGNIFICANCE_LEVEL: float = 0.05

    # ── Paths ─────────────────────────────────────────────────────
    RESULTS_DIR: str = "experiments/results"
    PLOTS_DIR: str = "experiments/results/plots"
    LOGS_DIR: str = "experiments/results/logs"
    TABLES_DIR: str = "experiments/results/tables"

    # ── Method names (canonical order) ────────────────────────────
    METHOD_NAMES: tuple[str, ...] = (
        "bagging", "random_forest", "adaboost", "gradient_boosting",
        "hard_voting", "soft_voting",
        "decision_tree", "logistic_regression",
    )

    # ── Default n_estimators for quick comparisons ────────────────
    DEFAULT_N_ESTIMATORS: int = 100


CFG = StudyConfig()


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (CFG.RESULTS_DIR, CFG.PLOTS_DIR, CFG.LOGS_DIR, CFG.TABLES_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def parse_ratio(ratio_str: str) -> float:
    """Parse '1:5' → 0.2 (minority/majority fraction)."""
    parts = ratio_str.split(":")
    return int(parts[0]) / int(parts[1])


if __name__ == "__main__":
    ensure_dirs()
    print(f"Seeds: {CFG.RANDOM_SEEDS}")
    print(f"N_RUNS: {CFG.N_RUNS}")
    print(f"Methods: {CFG.METHOD_NAMES}")
    print(f"parse_ratio('1:5') = {parse_ratio('1:5')}")
