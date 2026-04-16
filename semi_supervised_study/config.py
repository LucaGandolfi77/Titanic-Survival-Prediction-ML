"""
Central configuration for the Semi-Supervised Learning study.

All constants, seeds, experimental grids, and directory paths live here
in a frozen dataclass so that every module references the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class StudyConfig:
    # ── reproducibility ────────────────────────────────────────
    RANDOM_SEEDS: Tuple[int, ...] = (42, 7, 13, 99, 100, 21, 55, 77, 11, 33, 3, 17, 88, 44, 66)
    N_RUNS: int = 15
    CV_FOLDS: int = 5

    # ── experimental grid ──────────────────────────────────────
    LABELED_FRACTIONS: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    LABEL_NOISE_RATES: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    K_OVER_C_RATIOS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    ACTIVE_BUDGET_STEPS: Tuple[int, ...] = (5, 10, 20, 40, 60, 80, 100)
    ACTIVE_ROUNDS: int = 10

    # ── clustering defaults ────────────────────────────────────
    CLUSTER_ALGOS: Tuple[str, ...] = ("kmeans", "gmm", "agglomerative")
    DEFAULT_CLUSTER_ALGO: str = "kmeans"

    # ── classifier ─────────────────────────────────────────────
    RF_N_ESTIMATORS: int = 100

    # ── SSL strategy names ─────────────────────────────────────
    STRATEGY_NAMES: Tuple[str, ...] = (
        "supervised_baseline",
        "fully_supervised",
        "pseudo_labeling",
        "cluster_as_features",
        "cluster_prototypes",
        "active_learning",
        "label_propagation",
    )

    # ── datasets ───────────────────────────────────────────────
    REAL_DATASETS: Tuple[str, ...] = ("iris", "wine", "breast_cancer")
    SYNTH_DATASETS: Tuple[str, ...] = ("make_blobs", "make_classification")

    # ── alignment control ──────────────────────────────────────
    ALIGNMENT_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
    ALIGNMENT_NMI_TARGETS: Tuple[float, ...] = (0.2, 0.5, 0.8)

    # ── statistical tests ──────────────────────────────────────
    SIGNIFICANCE_LEVEL: float = 0.05

    # ── paths ──────────────────────────────────────────────────
    PROJECT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def RESULTS_DIR(self) -> Path:
        return self.PROJECT_DIR / "experiments" / "results"

    @property
    def PLOTS_DIR(self) -> Path:
        return self.RESULTS_DIR / "plots"

    @property
    def LOGS_DIR(self) -> Path:
        return self.RESULTS_DIR / "logs"

    @property
    def TABLES_DIR(self) -> Path:
        return self.RESULTS_DIR / "tables"


CFG = StudyConfig()


def ensure_dirs() -> None:
    for d in (CFG.RESULTS_DIR, CFG.PLOTS_DIR, CFG.LOGS_DIR, CFG.TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Seeds       : {len(CFG.RANDOM_SEEDS)}")
    print(f"Fractions   : {CFG.LABELED_FRACTIONS}")
    print(f"Strategies  : {CFG.STRATEGY_NAMES}")
    print(f"Results dir : {CFG.RESULTS_DIR}")
