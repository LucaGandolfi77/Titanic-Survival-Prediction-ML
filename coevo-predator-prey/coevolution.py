"""
coevolution.py — Co-evolutionary engine orchestrating both GP populations.

Biological analogy:
    This module implements the *Red Queen* dynamic: predators and prey
    continuously adapt to each other across generations, creating an
    evolutionary arms race where relative — not absolute — fitness drives
    selection.  The round-robin opponent sampling mirrors nature's
    frequency-dependent selection (an organism's success depends on the
    current composition of the opposing population).
"""

from __future__ import annotations

import random as py_random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import gp, tools

from config import SimConfig
from fitness import (
    EpisodeResult,
    evaluate_predator,
    evaluate_prey,
    run_episode,
)
from gp_setup import (
    create_predator_toolbox,
    create_prey_toolbox,
    get_predator_pset,
    get_prey_pset,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Per-generation statistics logged by the engine."""
    generation: int = 0
    prey_fitness_best: float = 0.0
    prey_fitness_mean: float = 0.0
    prey_fitness_std: float = 0.0
    pred_fitness_best: float = 0.0
    pred_fitness_mean: float = 0.0
    pred_fitness_std: float = 0.0
    avg_prey_survival: float = 0.0
    avg_prey_food: float = 0.0
    avg_pred_kills: float = 0.0
    avg_episode_duration: float = 0.0
    behaviors: List[str] = field(default_factory=list)
    best_prey_expr: str = ""
    best_pred_expr: str = ""


@dataclass
class CoevolutionResult:
    """Full result of a co-evolutionary run."""
    generation_stats: List[GenerationStats] = field(default_factory=list)
    best_prey_individual: Any = None
    best_predator_individual: Any = None
    config: Optional[SimConfig] = None
    sample_episodes: List[EpisodeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CoevolutionEngine:
    """Orchestrates two GP populations for N co-evolutionary generations.

    At each generation:
        1. Evaluate every prey individual against K sampled predators.
        2. Evaluate every predator individual against K sampled prey.
        3. Run behaviour analysis on a representative episode.
        4. Log statistics.
        5. Select, cross, mutate both populations.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._rng = np.random.Generator(np.random.PCG64(config.seed))

        # Seed Python's random (used by DEAP internally).
        py_random.seed(config.seed)

        # Toolboxes.
        self.prey_tb = create_prey_toolbox(config)
        self.pred_tb = create_predator_toolbox(config)

        # Populations.
        self.prey_pop: List[Any] = self.prey_tb.population(n=config.pop_size)
        self.pred_pop: List[Any] = self.pred_tb.population(n=config.pop_size)

        # Primitive sets (for expression rendering).
        self._prey_pset = get_prey_pset(config)
        self._pred_pset = get_predator_pset(config)

    # -- Public interface ----------------------------------------------------

    def run(
        self,
        behavior_callback: Optional[Any] = None,
        log_callback: Optional[Any] = None,
    ) -> CoevolutionResult:
        """Execute the full co-evolutionary loop.

        Args:
            behavior_callback: callable(EpisodeResult, SimConfig) -> list[str]
                               If provided, called each generation to detect
                               emergent behaviours.
            log_callback:      callable(GenerationStats) — called each gen
                               for logging.

        Returns:
            A ``CoevolutionResult`` with all statistics.
        """
        result = CoevolutionResult(config=self.config)
        cfg = self.config

        for gen in range(cfg.generations):
            stats = GenerationStats(generation=gen)

            # --- 1. Evaluate prey --------------------------------------------
            pred_sample_pool = self.pred_pop[:]
            for ind in self.prey_pop:
                k = min(cfg.opponent_sample_k, len(pred_sample_pool))
                opponents = _sample_k(pred_sample_pool, k, self._rng)
                ind.fitness.values = evaluate_prey(
                    individual=ind,
                    predator_sample=opponents,
                    config=cfg,
                    prey_compile=self.prey_tb.compile,
                    pred_compile=self.pred_tb.compile,
                    rng=self._rng,
                )

            # --- 2. Evaluate predators ---------------------------------------
            prey_sample_pool = self.prey_pop[:]
            for ind in self.pred_pop:
                k = min(cfg.opponent_sample_k, len(prey_sample_pool))
                opponents = _sample_k(prey_sample_pool, k, self._rng)
                ind.fitness.values = evaluate_predator(
                    individual=ind,
                    prey_sample=opponents,
                    config=cfg,
                    prey_compile=self.prey_tb.compile,
                    pred_compile=self.pred_tb.compile,
                    rng=self._rng,
                )

            # --- 3. Collect fitness stats ------------------------------------
            prey_fits = [ind.fitness.values[0] for ind in self.prey_pop]
            pred_fits = [ind.fitness.values[0] for ind in self.pred_pop]

            stats.prey_fitness_best = max(prey_fits) if prey_fits else 0.0
            stats.prey_fitness_mean = float(np.mean(prey_fits)) if prey_fits else 0.0
            stats.prey_fitness_std = float(np.std(prey_fits)) if prey_fits else 0.0
            stats.pred_fitness_best = max(pred_fits) if pred_fits else 0.0
            stats.pred_fitness_mean = float(np.mean(pred_fits)) if pred_fits else 0.0
            stats.pred_fitness_std = float(np.std(pred_fits)) if pred_fits else 0.0

            # --- 4. Run a representative episode for behaviour analysis ------
            best_prey = tools.selBest(self.prey_pop, 1)[0]
            best_pred = tools.selBest(self.pred_pop, 1)[0]

            prey_func = self.prey_tb.compile(expr=best_prey)
            pred_func = self.pred_tb.compile(expr=best_pred)

            ep_rng = np.random.Generator(
                np.random.PCG64(int(self._rng.integers(0, 2**31))))
            episode = run_episode(
                prey_funcs=[prey_func] * cfg.num_prey,
                predator_funcs=[pred_func] * cfg.num_predators,
                config=cfg,
                rng=ep_rng,
            )

            stats.avg_prey_survival = float(
                np.mean(episode.prey_survival_times)) if episode.prey_survival_times else 0.0
            stats.avg_prey_food = episode.total_prey_food / max(cfg.num_prey, 1)
            stats.avg_pred_kills = episode.total_predator_kills / max(cfg.num_predators, 1)
            stats.avg_episode_duration = float(episode.duration)

            # Behaviour detection.
            if behavior_callback is not None:
                try:
                    stats.behaviors = behavior_callback(episode, cfg)
                except Exception:
                    stats.behaviors = []

            # Best tree expressions.
            stats.best_prey_expr = str(best_prey)
            stats.best_pred_expr = str(best_pred)

            result.generation_stats.append(stats)

            # Store last-gen episode for visualization.
            if gen == cfg.generations - 1:
                result.sample_episodes.append(episode)

            # Log.
            if log_callback is not None:
                try:
                    log_callback(stats)
                except Exception:
                    pass

            # --- 5. Selection + variation ------------------------------------
            self.prey_pop = self._evolve_population(
                self.prey_pop, self.prey_tb, cfg)
            self.pred_pop = self._evolve_population(
                self.pred_pop, self.pred_tb, cfg)

        # Best individuals overall.
        result.best_prey_individual = tools.selBest(self.prey_pop, 1)[0]
        result.best_predator_individual = tools.selBest(self.pred_pop, 1)[0]

        return result

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _evolve_population(
        population: List[Any],
        toolbox: tools.Toolbox,
        config: SimConfig,
    ) -> List[Any]:
        """Tournament-select, crossover, and mutate a single population.

        Args:
            population: Current generation.
            toolbox:    DEAP toolbox for this species.
            config:     Simulation config.

        Returns:
            New population list of the same size.
        """
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover in pairs.
        for i in range(0, len(offspring) - 1, 2):
            if py_random.random() < config.crossover_prob:
                offspring[i], offspring[i + 1] = toolbox.mate(
                    offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation.
        for i in range(len(offspring)):
            if py_random.random() < config.mutation_prob:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _sample_k(
    pool: List[Any],
    k: int,
    rng: np.random.Generator,
) -> List[Any]:
    """Sample k distinct individuals from pool (without replacement).

    If the pool is smaller than k, returns the full pool.

    Args:
        pool: Population to sample from.
        k:    Sample size.
        rng:  Random generator.

    Returns:
        A list of up to k individuals.
    """
    if len(pool) <= k:
        return pool[:]
    indices = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in indices]
