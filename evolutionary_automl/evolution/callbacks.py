"""
Callbacks for evolutionary loops: logging, early stopping, statistics.

These hooks are called at the end of each generation to record progress,
check stopping criteria, and maintain the hall of fame.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from deap import tools

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    median_fitness: float
    diversity: float
    elapsed_time: float
    best_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "min_fitness": self.min_fitness,
            "median_fitness": self.median_fitness,
            "diversity": self.diversity,
            "elapsed_time": self.elapsed_time,
            "best_description": self.best_description,
        }


class EvolutionLogger:
    """Records per-generation statistics and saves them to JSON."""

    def __init__(self) -> None:
        self.history: List[GenerationStats] = []
        self._start_time = time.perf_counter()

    def record(
        self,
        generation: int,
        population: list,
        best_description: str = "",
    ) -> GenerationStats:
        """Compute and record statistics for a generation."""
        fitnesses = []
        for ind in population:
            if hasattr(ind, "fitness") and ind.fitness.valid:
                fitnesses.append(ind.fitness.values[0])

        if not fitnesses:
            fitnesses = [0.0]

        fit_arr = np.array(fitnesses)
        unique_ratio = len(set(tuple(ind) for ind in population)) / len(population)

        stats = GenerationStats(
            generation=generation,
            best_fitness=float(np.max(fit_arr)),
            mean_fitness=float(np.mean(fit_arr)),
            std_fitness=float(np.std(fit_arr)),
            min_fitness=float(np.min(fit_arr)),
            median_fitness=float(np.median(fit_arr)),
            diversity=unique_ratio,
            elapsed_time=time.perf_counter() - self._start_time,
            best_description=best_description,
        )
        self.history.append(stats)

        logger.info(
            f"Gen {generation:3d} | Best: {stats.best_fitness:.4f} | "
            f"Mean: {stats.mean_fitness:.4f} ± {stats.std_fitness:.4f} | "
            f"Diversity: {stats.diversity:.2%} | "
            f"Time: {stats.elapsed_time:.1f}s"
        )
        return stats

    def save(self, path: Path) -> None:
        """Save full history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self.history]
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Evolution log saved to {path}")

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.history]


class EarlyStopping:
    """Stop evolution if fitness hasn't improved for `patience` generations."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = -np.inf
        self._wait = 0

    def should_stop(self, best_fitness: float) -> bool:
        if best_fitness > self._best + self.min_delta:
            self._best = best_fitness
            self._wait = 0
            return False
        self._wait += 1
        if self._wait >= self.patience:
            logger.info(
                f"Early stopping triggered after {self._wait} generations "
                f"without improvement (best={self._best:.4f})"
            )
            return True
        return False


if __name__ == "__main__":
    ev_logger = EvolutionLogger()
    print("Logger initialized, ready to record generations.")
