"""
logging_utils.py — Generation-level CSV logging and console output.

Records per-generation statistics to a CSV file and prints a summary
to the console.  Also saves the best GP trees as string expressions
every *N* generations.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, Sequence

from deap import gp as deap_gp

from coevolution import GenerationStats


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append-mode CSV logger for generation statistics.

    Args:
        output_dir: Directory where ``generations.csv`` will be written.
    """

    HEADER = [
        "generation",
        "prey_fit_mean", "prey_fit_std", "prey_fit_best",
        "pred_fit_mean", "pred_fit_std", "pred_fit_best",
        "avg_prey_survival", "avg_prey_food", "avg_pred_kills",
        "behaviors",
    ]

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "generations.csv"
        self._file: Optional[IO[str]] = None
        self._writer: Optional[csv.writer] = None

    # Context-manager support
    def open(self) -> "CSVLogger":
        """Open the CSV file (truncating any previous content)."""
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        return self

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "CSVLogger":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def log(self, gen: int, stats: GenerationStats) -> None:
        """Write a single row to the CSV.

        Args:
            gen:   Generation index.
            stats: Collected statistics for this generation.
        """
        if self._writer is None:
            return
        self._writer.writerow([
            gen,
            f"{stats.prey_fitness_mean:.4f}",
            f"{stats.prey_fitness_std:.4f}",
            f"{stats.prey_fitness_best:.4f}",
            f"{stats.pred_fitness_mean:.4f}",
            f"{stats.pred_fitness_std:.4f}",
            f"{stats.pred_fitness_best:.4f}",
            f"{stats.avg_prey_survival:.2f}",
            f"{stats.avg_prey_food:.2f}",
            f"{stats.avg_pred_kills:.2f}",
            "|".join(stats.behaviors) if stats.behaviors else "",
        ])
        self._file.flush()


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def print_generation(gen: int, total: int, stats: GenerationStats) -> None:
    """Print a one-line console summary for a generation.

    Args:
        gen:   Current generation (0-based).
        total: Total number of generations.
        stats: Collected statistics.
    """
    beh = ", ".join(stats.behaviors) if stats.behaviors else "none"
    print(
        f"[Gen {gen + 1:>4}/{total}] "
        f"Prey fit={stats.prey_fitness_mean:6.1f}±{stats.prey_fitness_std:5.1f} "
        f"Pred fit={stats.pred_fitness_mean:6.1f}±{stats.pred_fitness_std:5.1f} "
        f"survived={stats.avg_prey_survival:4.1f} food={stats.avg_prey_food:4.1f} "
        f"kills={stats.avg_pred_kills:4.1f} | {beh}"
    )


# ---------------------------------------------------------------------------
# GP-tree expression saver
# ---------------------------------------------------------------------------

def save_best_trees(
    output_dir: str | Path,
    gen: int,
    best_prey,
    best_pred,
    prey_pset,
    pred_pset,
) -> None:
    """Persist the best GP tree expressions to a text file.

    Args:
        output_dir: Directory to write into.
        gen:        Generation index.
        best_prey:  Best prey individual (DEAP GP tree).
        best_pred:  Best predator individual (DEAP GP tree).
        prey_pset:  PrimitiveSet used for prey.
        pred_pset:  PrimitiveSet used for predators.
    """
    d = Path(output_dir) / "trees"
    d.mkdir(parents=True, exist_ok=True)

    path = d / f"gen_{gen:05d}.txt"
    with open(path, "w") as f:
        f.write(f"=== Generation {gen} ===\n\n")
        f.write(f"Best prey  (fitness={best_prey.fitness.values[0]:.4f}):\n")
        f.write(f"  {str(best_prey)}\n\n")
        f.write(f"Best pred  (fitness={best_pred.fitness.values[0]:.4f}):\n")
        f.write(f"  {str(best_pred)}\n")
