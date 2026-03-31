"""
Configuration — Decision Tree Study
====================================
Central configuration for all experiments. Every tunable constant lives here
so that experiments are reproducible and easy to reconfigure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class StudyConfig:
    """Immutable project-wide configuration."""

    # Reproducibility
    RANDOM_SEEDS: List[int] = field(
        default_factory=lambda: [42, 7, 13, 99, 100, 21, 55, 77, 11, 33]
    )
    N_RUNS: int = 10
    CV_FOLDS: int = 10

    # Depth sweep
    DEPTHS: List[Optional[int]] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
    )

    # Dataset size sweep
    DATASET_SIZES: List[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000]
    )

    # Noise parameters
    LABEL_NOISE_RATES: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.30]
    )
    FEATURE_NOISE_SIGMAS: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 1.0]
    )

    # Pre-pruning sweeps
    MIN_SAMPLES_LEAF_VALUES: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50]
    )
    MIN_SAMPLES_SPLIT_VALUES: List[int] = field(
        default_factory=lambda: [2, 5, 10, 20]
    )

    # Cost-complexity pruning
    ALPHA_N_STEPS: int = 30

    # Pruning strategy names
    PRUNING_STRATEGIES: List[str] = field(
        default_factory=lambda: [
            "none", "pre_depth", "pre_samples", "ccp", "combined"
        ]
    )

    # Statistical testing
    SIGNIFICANCE_LEVEL: float = 0.05

    # Paths
    RESULTS_DIR: Path = Path("experiments/results")
    PLOTS_DIR: Path = Path("experiments/results/plots")
    LOGS_DIR: Path = Path("experiments/results/logs")
    TABLES_DIR: Path = Path("experiments/results/tables")

    # Plotting
    FIGURE_DPI: int = 150
    FIGURE_FORMAT: str = "png"
    COLOR_PALETTE: str = "colorblind"


CFG = StudyConfig()


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (CFG.RESULTS_DIR, CFG.PLOTS_DIR, CFG.LOGS_DIR, CFG.TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Seeds: {CFG.RANDOM_SEEDS}")
    print(f"Depths: {CFG.DEPTHS}")
    print(f"Label noise rates: {CFG.LABEL_NOISE_RATES}")
    print(f"Results dir: {CFG.RESULTS_DIR}")
