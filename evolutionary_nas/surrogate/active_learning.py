"""
Active Learning
===============
Select architectures with highest uncertainty for real evaluation,
balancing exploitation (high predicted accuracy) and exploration
(high uncertainty).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from surrogate.predictor import SurrogatePredictor


def select_for_real_eval(
    surrogate: SurrogatePredictor,
    genomes: List[List[float]],
    net_type: str,
    top_k_frac: float = 0.3,
    exploration_weight: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Select which genomes should be evaluated with real training
    vs. which can rely on surrogate prediction.

    Uses an acquisition function: score = predicted_acc + w * uncertainty.

    Returns:
        (real_eval_indices, surrogate_only_indices)
    """
    if not surrogate.is_fitted or len(genomes) == 0:
        return list(range(len(genomes))), []

    predictions = surrogate.predict_batch(genomes, net_type)
    uncertainties = surrogate.uncertainty_batch(genomes, net_type)

    acquisition = predictions + exploration_weight * uncertainties

    n_real = max(1, int(len(genomes) * top_k_frac))
    ranked = np.argsort(acquisition)[::-1]

    real_indices = ranked[:n_real].tolist()
    surrogate_indices = ranked[n_real:].tolist()

    return real_indices, surrogate_indices


if __name__ == "__main__":
    from search_space.mlp_space import random_mlp_genome
    rng = np.random.default_rng(42)
    surrogate = SurrogatePredictor()
    genomes = [random_mlp_genome(rng) for _ in range(50)]
    for g in genomes[:30]:
        surrogate.add_observation(g, "mlp", rng.uniform(0.5, 0.99))
    surrogate.fit()
    real_idx, surr_idx = select_for_real_eval(surrogate, genomes, "mlp", 0.3)
    print(f"Real evals: {len(real_idx)}, Surrogate-only: {len(surr_idx)}")
