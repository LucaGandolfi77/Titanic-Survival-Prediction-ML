"""
Evolution Callbacks
===================
Per-generation logging, diversity tracking, hall-of-fame management,
and convergence detection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from deap import tools

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    generation: int = 0
    best_fitness: float = 0.0
    mean_fitness: float = 0.0
    std_fitness: float = 0.0
    diversity_index: float = 0.0
    best_description: str = ""
    surrogate_spearman_rho: float = 0.0
    n_real_evals: int = 0
    n_surrogate_evals: int = 0
    elapsed_seconds: float = 0.0


class EvolutionLogger:
    """Records statistics per generation and saves to JSON."""

    def __init__(self, log_path: Optional[Path] = None):
        self.history: List[GenerationStats] = []
        self.log_path = log_path
        self._start = time.perf_counter()

    def record(
        self,
        generation: int,
        fitnesses: List[float],
        best_desc: str = "",
        surrogate_rho: float = 0.0,
        n_real: int = 0,
        n_surr: int = 0,
        population: Optional[List[List[float]]] = None,
    ) -> GenerationStats:
        fits = np.array(fitnesses) if fitnesses else np.array([0.0])
        diversity = 0.0
        if population is not None:
            unique = set(tuple(round(g, 4) for g in ind) for ind in population)
            diversity = len(unique) / max(len(population), 1)

        stats = GenerationStats(
            generation=generation,
            best_fitness=float(np.max(fits)),
            mean_fitness=float(np.mean(fits)),
            std_fitness=float(np.std(fits)),
            diversity_index=diversity,
            best_description=best_desc,
            surrogate_spearman_rho=surrogate_rho,
            n_real_evals=n_real,
            n_surrogate_evals=n_surr,
            elapsed_seconds=time.perf_counter() - self._start,
        )
        self.history.append(stats)
        logger.info(
            f"Gen {generation:3d} | best={stats.best_fitness:.4f} "
            f"mean={stats.mean_fitness:.4f}±{stats.std_fitness:.4f} "
            f"div={stats.diversity_index:.2f} ρ={stats.surrogate_spearman_rho:.3f}"
        )
        return stats

    def save(self, path: Optional[Path] = None) -> None:
        p = path or self.log_path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in self.history]
        p.write_text(json.dumps(data, indent=2))

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(s) for s in self.history]


class EarlyStopping:
    """Stop evolution if best fitness does not improve for `patience` generations."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = -float("inf")
        self._counter = 0

    def should_stop(self, best_fitness: float) -> bool:
        if best_fitness > self._best + self.min_delta:
            self._best = best_fitness
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience

    def reset(self) -> None:
        self._best = -float("inf")
        self._counter = 0


if __name__ == "__main__":
    elog = EvolutionLogger()
    for g in range(5):
        fits = np.random.uniform(0.5, 0.95, size=50).tolist()
        elog.record(g, fits, "test arch", population=[[0.0]*14]*50)
    print(json.dumps(elog.to_dict()[:2], indent=2))
