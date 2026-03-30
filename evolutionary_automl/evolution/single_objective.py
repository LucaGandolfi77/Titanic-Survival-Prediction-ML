"""
Single-objective Genetic Algorithm using DEAP.

Maximizes macro-averaged F1 score via tournament selection, two-point
typed crossover, mixed-type mutation, and elitism (Hall of Fame).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from ..config import CFG
from ..fitness.cache import FitnessCache
from ..fitness.evaluator import FitnessEvaluator
from ..genome.chromosome import CHROMOSOME_LENGTH, chromosome_description
from ..genome.initializer import seeded_population
from ..genome.operators import cx_two_point_typed, mut_mixed_type
from ..search_space.validators import repair_chromosome
from .callbacks import EarlyStopping, EvolutionLogger

logger = logging.getLogger(__name__)


def _ensure_creator() -> None:
    """Register DEAP creator types once."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def run_single_objective_ga(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    pop_size: int = CFG.POP_SIZE_GA,
    n_gen: int = CFG.N_GEN_GA,
    cx_pb: float = CFG.CX_PB,
    mut_pb: float = CFG.MUT_PB,
    tournament_size: int = CFG.TOURNAMENT_SIZE,
    elite_size: int = CFG.ELITE_SIZE,
    seed: int = 42,
    early_stopping_patience: int = 10,
    cache: Optional[FitnessCache] = None,
) -> Dict[str, Any]:
    """Run single-objective GA and return results dict.

    Returns:
        Dictionary with keys: best_individual, best_fitness, history, hof,
        population, cache_stats, n_features.
    """
    _ensure_creator()

    random.seed(seed)
    np.random.seed(seed)

    evaluator = FitnessEvaluator(
        X, y,
        dataset_name=dataset_name,
        cache=cache if cache is not None else FitnessCache(),
        random_state=seed,
    )
    n_features = X.shape[1]

    toolbox = base.Toolbox()
    init_pop = seeded_population(pop_size, seed=seed)

    def _init_individual():
        return creator.Individual(init_pop.pop(0))

    toolbox.register("individual", _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate_single_objective)
    toolbox.register("mate", cx_two_point_typed)
    toolbox.register("mutate", mut_mixed_type, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(elite_size)
    ev_logger = EvolutionLogger()
    early_stop = EarlyStopping(patience=early_stopping_patience)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    desc = chromosome_description(hof[0], n_features)
    ev_logger.record(0, pop, best_description=desc)

    for gen in range(1, n_gen + 1):
        # Selection + variation
        offspring = toolbox.select(pop, len(pop) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_pb:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mut_pb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Repair
        for ind in offspring:
            repaired = repair_chromosome(list(ind))
            ind[:] = repaired

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Elitism: inject Hall of Fame into population
        elite = [toolbox.clone(e) for e in hof]
        pop[:] = elite + offspring

        hof.update(pop)
        desc = chromosome_description(hof[0], n_features)
        stats = ev_logger.record(gen, pop, best_description=desc)

        if early_stop.should_stop(stats.best_fitness):
            logger.info(f"Early stopping at generation {gen}")
            break

    best = hof[0]
    return {
        "best_individual": list(best),
        "best_fitness": best.fitness.values[0],
        "best_description": chromosome_description(best, n_features),
        "history": ev_logger.to_dict_list(),
        "hof": [list(ind) for ind in hof],
        "eval_count": evaluator.eval_count,
        "cache_stats": repr(evaluator.cache),
        "n_features": n_features,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    results = run_single_objective_ga(
        X, y, dataset_name="iris", pop_size=20, n_gen=5, seed=42
    )
    print(f"\nBest F1: {results['best_fitness']:.4f}")
    print(f"Best pipeline: {results['best_description']}")
    print(f"Total evaluations: {results['eval_count']}")
