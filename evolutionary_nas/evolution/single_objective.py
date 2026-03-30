"""
Single-Objective GA
===================
DEAP-based genetic algorithm maximizing a composite fitness:
    fitness = accuracy − λ · log₁₀(param_count)

Uses tournament selection, gene-type-aware crossover/mutation,
elitism via Hall of Fame, and optional surrogate acceleration.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from config import CFG, set_seed
from search_space.genome_encoder import describe, repair
from fitness.evaluator import FitnessEvaluator
from evolution.operators import cx_two_point_typed, mut_mixed_type
from evolution.initializer import biased_small_population
from evolution.callbacks import EvolutionLogger, EarlyStopping
from surrogate.surrogate_trainer import SurrogateTrainer

logger = logging.getLogger(__name__)

# DEAP creator setup (idempotent)
if not hasattr(creator, "FitnessMaxSO"):
    creator.create("FitnessMaxSO", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualSO"):
    creator.create("IndividualSO", list, fitness=creator.FitnessMaxSO)


def run_single_objective_ga(
    evaluator: FitnessEvaluator,
    net_type: str,
    pop_size: int = 50,
    n_gen: int = 40,
    cxpb: float = 0.7,
    mutpb: float = 0.15,
    lambda_penalty: float = 0.5,
    seed: int = 42,
    use_surrogate: bool = True,
    surrogate_trainer: Optional[SurrogateTrainer] = None,
    log_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a single-objective GA and return results dictionary."""
    set_seed(seed)

    toolbox = base.Toolbox()
    rng = np.random.default_rng(seed)

    # Population initialization
    init_pop = biased_small_population(pop_size, net_type, rng)
    population = [creator.IndividualSO(g) for g in init_pop]

    toolbox.register("mate", cx_two_point_typed, net_type=net_type)
    toolbox.register("mutate", mut_mixed_type, net_type=net_type, indpb=mutpb)
    toolbox.register("select", tools.selTournament, tournsize=CFG.TOURNAMENT_SIZE)

    hof = tools.HallOfFame(CFG.HOF_SIZE)
    evo_logger = EvolutionLogger(log_path=log_path)
    early_stop = EarlyStopping(patience=CFG.EARLY_STOP_PATIENCE)

    st = surrogate_trainer
    if use_surrogate and st is None:
        st = SurrogateTrainer(
            net_type, CFG.SURROGATE_WARMUP, CFG.SURROGATE_TOPK, CFG.SURROGATE_RETRAIN_EVERY
        )

    # Evaluate initial population
    for ind in population:
        fitness = evaluator.evaluate_single_objective(ind, lambda_penalty)
        ind.fitness.values = fitness
        if st is not None:
            acc, _ = evaluator.evaluate(ind)
            st.record_evaluation(list(ind), acc)

    hof.update(population)

    for gen in range(n_gen):
        # Selection + variation
        offspring = toolbox.select(population, len(population) - CFG.HOF_SIZE)
        offspring = [creator.IndividualSO(list(ind)) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Repair
        for ind in offspring:
            ind[:] = repair(list(ind), net_type)

        # Determine which need real evaluation
        invalid = [ind for ind in offspring if not ind.fitness.valid]

        if st is not None and st.is_warmed_up and st.predictor.is_fitted:
            real_idx, surr_idx = st.select_candidates([list(i) for i in invalid])

            # Surrogate-only evaluations
            for idx in surr_idx:
                pred_acc = st.predict(list(invalid[idx]))
                invalid[idx].fitness.values = (pred_acc,)

            # Real evaluations
            for idx in real_idx:
                fitness = evaluator.evaluate_single_objective(
                    list(invalid[idx]), lambda_penalty
                )
                invalid[idx].fitness.values = fitness
                acc, _ = evaluator.evaluate(list(invalid[idx]))
                st.record_evaluation(list(invalid[idx]), acc)
        else:
            for ind in invalid:
                fitness = evaluator.evaluate_single_objective(
                    list(ind), lambda_penalty
                )
                ind.fitness.values = fitness
                if st is not None:
                    acc, _ = evaluator.evaluate(list(ind))
                    st.record_evaluation(list(ind), acc)

        # Retrain surrogate
        if st is not None and st.should_retrain(gen):
            st.retrain()

        # Elitism: reinsert HoF
        elites = [creator.IndividualSO(list(h)) for h in hof]
        for e in elites:
            e.fitness.values = evaluator.evaluate_single_objective(
                list(e), lambda_penalty
            )
        population = offspring + elites

        hof.update(population)

        # Log
        fits = [ind.fitness.values[0] for ind in population]
        best_desc = describe(list(hof[0]), net_type) if hof else ""
        surr_rho = (st.predictor.spearman_history[-1]
                    if st and st.predictor.spearman_history else 0.0)
        n_real = st._n_real_evals if st else len(invalid)
        n_surr = st._n_surrogate_evals if st else 0

        evo_logger.record(
            gen, fits, best_desc, surr_rho, n_real, n_surr,
            population=[list(i) for i in population],
        )

        if early_stop.should_stop(max(fits)):
            logger.info(f"Early stopping at generation {gen}")
            break

    if log_path:
        evo_logger.save()

    best = list(hof[0]) if hof else population[0]
    return {
        "best_genome": best,
        "best_fitness": float(hof[0].fitness.values[0]) if hof else 0.0,
        "best_description": describe(best, net_type),
        "history": evo_logger.to_dict(),
        "hof": [list(h) for h in hof],
        "cache_stats": {
            "hits": evaluator.cache.hits,
            "misses": evaluator.cache.misses,
            "hit_rate": evaluator.cache.hit_rate,
        },
        "surrogate_stats": st.stats if st else None,
    }


if __name__ == "__main__":
    print("Single-objective GA module ready.")
