"""
Configuration — Adaptive Clustering Study
============================================
Central repository for all experimental constants, paths, and
algorithm hyperparameters.  Frozen dataclass ensures values cannot
be accidentally mutated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    # ── Seeds & repetitions ───────────────────────────────────────
    RANDOM_SEEDS: tuple[int, ...] = (
        42, 7, 13, 99, 100, 21, 55, 77, 11, 33, 3, 17, 88, 44, 66,
    )
    N_RUNS: int = 15

    # ── K-range for cluster-number experiments ────────────────────
    K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15)
    DEFAULT_K: int = 5

    # ── Dataset parameters ────────────────────────────────────────
    SYNTH_N_SAMPLES: list[int] = field(
        default_factory=lambda: [200, 500, 1000, 2000, 5000],
    )
    SYNTH_N_FEATURES: int = 2
    NOISE_LEVELS: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.30],
    )
    OUTLIER_FRACTIONS: list[float] = field(
        default_factory=lambda: [0.0, 0.02, 0.05, 0.10, 0.15],
    )
    HIGH_DIM_FEATURES: list[int] = field(
        default_factory=lambda: [2, 5, 10, 20, 50, 100],
    )

    # ── ISODATA parameters ────────────────────────────────────────
    ISODATA_K_INIT: int = 8
    ISODATA_THETA_N: int = 10       # min samples per cluster
    ISODATA_THETA_S: float = 1.0    # max intra-cluster std (split)
    ISODATA_THETA_C: float = 1.0    # min inter-centroid dist (merge)
    ISODATA_MAX_MERGE: int = 2      # max pairs to merge per iter
    ISODATA_MAX_ITER: int = 50

    # ── Adaptive framework parameters ─────────────────────────────
    ADAPT_K_MIN: int = 2
    ADAPT_K_MAX: int = 20
    ADAPT_SPLIT_SILHOUETTE: float = 0.15   # split if cluster sil < this
    ADAPT_MERGE_DISTANCE: float = 0.5      # merge if centroid dist < this
    ADAPT_STABILITY_THRESHOLD: float = 0.85
    ADAPT_MAX_ITER: int = 30
    ADAPT_N_BOOTSTRAP: int = 10

    # ── Algorithm suite ───────────────────────────────────────────
    METHOD_NAMES: tuple[str, ...] = (
        "kmeans", "isodata", "minibatch_kmeans",
        "bisecting_kmeans", "gmm", "adaptive",
    )

    # ── Paths ─────────────────────────────────────────────────────
    RESULTS_DIR: str = "experiments/results"
    PLOTS_DIR: str = "experiments/results/plots"
    LOGS_DIR: str = "experiments/results/logs"
    TABLES_DIR: str = "experiments/results/tables"

    # ── Statistical testing ───────────────────────────────────────
    SIGNIFICANCE_LEVEL: float = 0.05


CFG = StudyConfig()


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (CFG.RESULTS_DIR, CFG.PLOTS_DIR, CFG.LOGS_DIR, CFG.TABLES_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Methods: {CFG.METHOD_NAMES}")
    print(f"K range: {CFG.K_RANGE}")
