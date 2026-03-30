"""
Centralized configuration for the Evolutionary AutoML project.

All experimental parameters, paths, and random seeds are defined here
to ensure reproducibility and easy modification across experiments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Config:
    """Immutable experiment configuration."""

    # Reproducibility
    RANDOM_SEEDS: tuple = (42, 7, 13, 99, 100, 21, 55, 77, 11, 33)
    N_RUNS: int = 10

    # Single-objective GA
    POP_SIZE_GA: int = 50
    N_GEN_GA: int = 30
    CX_PB: float = 0.7
    MUT_PB: float = 0.2
    TOURNAMENT_SIZE: int = 3
    ELITE_SIZE: int = 5

    # Multi-objective NSGA-II
    POP_SIZE_NSGA: int = 100
    N_GEN_NSGA: int = 40

    # Fitness evaluation
    CV_FOLDS: int = 5
    MAX_EVAL_SECONDS: int = 60

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    RESULTS_DIR: Path = Path(__file__).resolve().parent / "experiments" / "results"
    PLOTS_DIR: Path = Path(__file__).resolve().parent / "experiments" / "results" / "plots"
    LOGS_DIR: Path = Path(__file__).resolve().parent / "experiments" / "results" / "logs"
    TABLES_DIR: Path = Path(__file__).resolve().parent / "experiments" / "results" / "tables"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Datasets
    DATASET_NAMES: tuple = ("iris", "breast_cancer", "wine", "digits")

    # Chromosome
    CHROMOSOME_LENGTH: int = 13

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (self.RESULTS_DIR, self.PLOTS_DIR, self.LOGS_DIR, self.TABLES_DIR):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()


if __name__ == "__main__":
    CFG.ensure_dirs()
    print(f"Project root : {CFG.PROJECT_ROOT}")
    print(f"Results dir  : {CFG.RESULTS_DIR}")
    print(f"Seeds        : {CFG.RANDOM_SEEDS}")
