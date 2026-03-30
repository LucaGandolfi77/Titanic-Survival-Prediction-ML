"""
Surrogate Trainer
=================
Manages the lifecycle of the surrogate model: warm-up on the initial
population, incremental updates, and periodic retraining.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from config import CFG
from surrogate.predictor import SurrogatePredictor
from surrogate.active_learning import select_for_real_eval

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """Orchestrates the surrogate model lifecycle during evolution."""

    def __init__(
        self,
        net_type: str,
        warmup_count: int = 50,
        top_k_frac: float = 0.3,
        retrain_every: int = 5,
    ):
        self.net_type = net_type
        self.warmup_count = warmup_count
        self.top_k_frac = top_k_frac
        self.retrain_every = retrain_every
        self.predictor = SurrogatePredictor()
        self._eval_count = 0
        self._generation = 0
        self._n_surrogate_evals = 0
        self._n_real_evals = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._eval_count >= self.warmup_count

    def record_evaluation(
        self, genome: List[float], accuracy: float
    ) -> None:
        """Record a real evaluation result for surrogate training."""
        self.predictor.add_observation(genome, self.net_type, accuracy)
        self._eval_count += 1

    def should_retrain(self, generation: int) -> bool:
        return (
            self.is_warmed_up and
            generation % self.retrain_every == 0
        )

    def retrain(self) -> float:
        """Retrain surrogate on all accumulated data. Returns Spearman ρ."""
        return self.predictor.fit()

    def select_candidates(
        self, genomes: List[List[float]]
    ) -> Tuple[List[int], List[int]]:
        """Decide which genomes need real evaluation vs surrogate-only."""
        if not self.is_warmed_up or not self.predictor.is_fitted:
            self._n_real_evals += len(genomes)
            return list(range(len(genomes))), []

        real_idx, surr_idx = select_for_real_eval(
            self.predictor, genomes, self.net_type, self.top_k_frac
        )
        self._n_real_evals += len(real_idx)
        self._n_surrogate_evals += len(surr_idx)
        return real_idx, surr_idx

    def predict(self, genome: List[float]) -> float:
        """Predict accuracy for a genome using the surrogate."""
        return self.predictor.predict(genome, self.net_type)

    def predict_batch(self, genomes: List[List[float]]) -> list:
        """Predict accuracy for multiple genomes."""
        preds = self.predictor.predict_batch(genomes, self.net_type)
        return preds.tolist()

    @property
    def stats(self) -> dict:
        return {
            "n_observations": self.predictor.n_observations,
            "is_warmed_up": self.is_warmed_up,
            "n_real_evals": self._n_real_evals,
            "n_surrogate_evals": self._n_surrogate_evals,
            "spearman_history": self.predictor.spearman_history,
        }


if __name__ == "__main__":
    st = SurrogateTrainer("mlp", warmup_count=10)
    print(f"Warmed up: {st.is_warmed_up}")
    import numpy as np
    from search_space.mlp_space import random_mlp_genome
    rng = np.random.default_rng(42)
    for _ in range(15):
        g = random_mlp_genome(rng)
        st.record_evaluation(g, rng.uniform(0.5, 0.99))
    rho = st.retrain()
    print(f"Warmed up: {st.is_warmed_up}, Spearman ρ: {rho:.4f}")
