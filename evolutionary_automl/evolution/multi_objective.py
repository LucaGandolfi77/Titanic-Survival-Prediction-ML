"""
Multi-objective NSGA-II evolutionary strategy using DEAP.

Simultaneously maximizes F1 macro and minimizes training time (or number
of features). Returns the full Pareto front of non-dominated solutions.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from ..config import CFG
from ..fitness.cache import FitnessCache
from ..fitness.evaluator import FitnessEvaluator
from ..genome.chromosome import CHROMOSOME_LENGTH, chromosome_description
from ..genome.initializer import seeded_population
from ..genome.operators import cx_two_point_typed, mut_mixed_type
from ..search_space.validators import repair_chromosome
from .callbacks import EvolutionLogger

logger = logging.getLogger(__name__)


def _ensure_creator_mo() -> None:
    """Register DEAP creator types for multi-objective optimization."""
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndividualMO"):
        creator.create("IndividualMO", list, fitness=creator.FitnessMulti)


def run_nsga2(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    pop_size: int = CFG.POP_SIZE_NSGA,
    n_gen: int = CFG.N_GEN_NSGA,
    cx_pb: float = CFG.CX_PB,
    mut_pb: float = CFG.MUT_PB,
    seed: int = 42,
    cache: Optional[FitnessCache] = None,
) -> Dict[str, Any]:
    """Run NSGA-II multi-objective optimization.

    Objectives: maximize F1 macro, minimize training time.

    Returns:
        Dictionary with: pareto_front, best_f1_individual, history, eval_count.
    """
    _ensure_creator_mo()

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
        return creator.IndividualMO(init_pop.pop(0))

    toolbox.register("individual", _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate_multi_objective)
    toolbox.register("mate", cx_two_point_typed)
    toolbox.register("mutate", mut_mixed_type, indpb=0.15)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    ev_logger = EvolutionLogger()

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    ev_logger.record(0, pop)

    for gen in range(1, n_gen + 1):
        # Assign crowding distance before tournament selection
        pop = toolbox.select(pop, len(pop))

        # Create offspring via selection + variation
        offspring = tools.selTournamentDCD(pop, len(pop))
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

        # Evaluate invalid
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # NSGA-II environmental selection
        pop = toolbox.select(pop + offspring, pop_size)

        ev_logger.record(gen, pop)

    # Extract Pareto front
    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto_data = []
    for ind in pareto:
        pareto_data.append({
            "chromosome": list(ind),
            "f1": ind.fitness.values[0],
            "training_time": ind.fitness.values[1],
            "description": chromosome_description(ind, n_features),
        })

    pareto_data.sort(key=lambda x: x["f1"], reverse=True)
    best_f1 = pareto_data[0] if pareto_data else None

    return {
        "pareto_front": pareto_data,
        "best_f1_individual": best_f1,
        "history": ev_logger.to_dict_list(),
        "eval_count": evaluator.eval_count,
        "cache_stats": repr(evaluator.cache),
        "n_features": n_features,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    results = run_nsga2(
        X, y, dataset_name="iris", pop_size=30, n_gen=5, seed=42
    )
    print(f"\nPareto front size: {len(results['pareto_front'])}")
    if results["best_f1_individual"]:
        best = results["best_f1_individual"]
        print(f"Best F1: {best['f1']:.4f}, Time: {best['training_time']:.4f}s")
        print(f"Pipeline: {best['description']}")
