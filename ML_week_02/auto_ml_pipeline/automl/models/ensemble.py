"""
Ensemble building  (Stage 5).

hard_voting   → VotingClassifier(voting='hard')
soft_voting   → VotingClassifier(voting='soft')
stacking      → StackingClassifier with LogisticRegression meta-learner
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleBuilder:
    """Build an ensemble from optimised models."""

    def __init__(self, config) -> None:
        self.method: str = getattr(config, "method", "stacking")
        self.meta_learner_cls = getattr(config, "meta_learner", "logistic_regression")

    def build(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task: str,
    ) -> Any:
        """Build & fit the ensemble. Returns a fitted sklearn estimator."""
        estimators = list(models.items())
        logger.info(f"Building {self.method} ensemble from {[n for n, _ in estimators]}")

        if self.method in ("hard_voting", "soft_voting"):
            voting = "soft" if self.method == "soft_voting" else "hard"
            if task == "classification":
                ens = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
            else:
                ens = VotingRegressor(estimators=estimators, n_jobs=-1)
        elif self.method == "stacking":
            meta = self._build_meta(task)
            if task == "classification":
                ens = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta,
                    cv=5,
                    n_jobs=-1,
                    passthrough=False,
                )
            else:
                ens = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta,
                    cv=5,
                    n_jobs=-1,
                    passthrough=False,
                )
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        ens.fit(X, y)
        logger.info("Ensemble fitted successfully.")
        return ens

    def _build_meta(self, task: str) -> Any:
        if task == "classification":
            return LogisticRegression(max_iter=1000)
        return Ridge()
