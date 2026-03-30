"""
Multi-Objective NSGA-II
=======================
DEAP-based NSGA-II: maximize accuracy, minimize parameter count.
Produces a Pareto front of non-dominated architectures.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

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

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
if not hasattr(creator, "IndividualMO"):
    creator.create("IndividualMO", list, fitness=creator.FitnessMulti)


def run_nsga2(
    evaluator: FitnessEvaluator,
    net_type: str,
    pop_size: int = 100,
    n_gen: int = 50,
    cxpb: float = 0.7,
    mutpb: float = 0.15,
    seed: int = 42,
    use_surrogate: bool = True,
    surrogate_trainer: Optional[SurrogateTrainer] = None,
    log_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run NSGA-II and return results with Pareto front."""
    set_seed(seed)

    toolbox = base.Toolbox()
    rng = np.random.default_rng(seed)

    init_pop = biased_small_population(pop_size, net_type, rng)
    population = [creator.IndividualMO(g) for g in init_pop]

    toolbox.register("mate", cx_two_point_typed, net_type=net_type)
    toolbox.register("mutate", mut_mixed_type, net_type=net_type, indpb=mutpb)
    toolbox.register("select", tools.selNSGA2)

    evo_logger = EvolutionLogger(log_path=log_path)

    st = surrogate_trainer
    if use_surrogate and st is None:
        st = SurrogateTrainer(
            net_type, CFG.SURROGATE_WARMUP, CFG.SURROGATE_TOPK, CFG.SURROGATE_RETRAIN_EVERY
        )

    # Evaluate initial population
    for ind in population:
        fitness = evaluator.evaluate_multi_objective(list(ind))
        ind.fitness.values = fitness
        if st is not None:
            acc, _ = evaluator.evaluate(list(ind))
            st.record_evaluation(list(ind), acc)

    for gen in range(n_gen):
        # Create offspring via tournament DCD
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [creator.IndividualMO(list(ind)) for ind in offspring]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        for ind in offspring:
            ind[:] = repair(list(ind), net_type)

        invalid = [ind for ind in offspring if not ind.fitness.valid]

        if st is not None and st.is_warmed_up and st.predictor.is_fitted:
            real_idx, surr_idx = st.select_candidates(
                [list(i) for i in invalid]
            )
            for idx in surr_idx:
                pred_acc = st.predict(list(invalid[idx]))
                # Use surrogate only for accuracy; params from actual config
                acc_real, params_real = evaluator.evaluate(list(invalid[idx]))
                invalid[idx].fitness.values = (pred_acc, -params_real)

            for idx in real_idx:
                fitness = evaluator.evaluate_multi_objective(list(invalid[idx]))
                invalid[idx].fitness.values = fitness
                acc, _ = evaluator.evaluate(list(invalid[idx]))
                st.record_evaluation(list(invalid[idx]), acc)
        else:
            for ind in invalid:
                fitness = evaluator.evaluate_multi_objective(list(ind))
                ind.fitness.values = fitness
                if st is not None:
                    acc, _ = evaluator.evaluate(list(ind))
                    st.record_evaluation(list(ind), acc)

        if st is not None and st.should_retrain(gen):
            st.retrain()

        # Combine parent + offspring, select pop_size
        combined = population + offspring
        population = toolbox.select(combined, pop_size)

        fits = [ind.fitness.values[0] for ind in population]
        surr_rho = (st.predictor.spearman_history[-1]
                    if st and st.predictor.spearman_history else 0.0)

        evo_logger.record(
            gen, fits, "", surr_rho,
            n_real=st._n_real_evals if st else len(invalid),
            n_surr=st._n_surrogate_evals if st else 0,
            population=[list(i) for i in population],
        )

    if log_path:
        evo_logger.save()

    # Extract Pareto front
    fronts = tools.sortNondominated(population, len(population), first_front_only=True)
    pareto_front = fronts[0] if fronts else []

    pareto_data = []
    for ind in pareto_front:
        acc, neg_params = ind.fitness.values
        pareto_data.append({
            "genome": list(ind),
            "accuracy": acc,
            "param_count": -neg_params,
            "description": describe(list(ind), net_type),
        })

    return {
        "pareto_front": pareto_data,
        "history": evo_logger.to_dict(),
        "surrogate_stats": st.stats if st else None,
    }


if __name__ == "__main__":
    print("NSGA-II module ready.")
