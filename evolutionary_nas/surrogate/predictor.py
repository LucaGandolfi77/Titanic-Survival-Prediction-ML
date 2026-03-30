"""
Surrogate Predictor
===================
XGBoost regressor that predicts architecture accuracy from genome features.
Provides uncertainty estimation via quantile regression for active learning.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from surrogate.feature_extractor import genome_to_features

logger = logging.getLogger(__name__)


class SurrogatePredictor:
    """XGBoost-based surrogate model for predicting architecture performance."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
        )
        self._model_upper = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
        )
        self._fitted = False
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._spearman_history: List[float] = []

    def add_observation(
        self, genome: List[float], net_type: str, accuracy: float
    ) -> None:
        """Record a (genome, accuracy) pair for training."""
        features = genome_to_features(genome, net_type)
        self._X.append(features)
        self._y.append(accuracy)

    def fit(self) -> float:
        """Train the surrogate on all accumulated data.

        Returns the in-sample Spearman correlation.
        """
        if len(self._X) < 5:
            return 0.0

        X = np.stack(self._X)
        y = np.array(self._y)

        self._model.fit(X, y)

        # Train an "upper bound" model on y + residuals for uncertainty
        preds = self._model.predict(X)
        residuals = np.abs(y - preds)
        self._model_upper.fit(X, y + residuals)

        self._fitted = True

        rho, _ = spearmanr(preds, y)
        rho = float(rho) if not np.isnan(rho) else 0.0
        self._spearman_history.append(rho)
        logger.info(f"Surrogate fitted on {len(y)} samples, Spearman ρ={rho:.4f}")
        return rho

    def predict(self, genome: List[float], net_type: str) -> float:
        """Predict accuracy for a single genome."""
        if not self._fitted:
            return 0.0
        features = genome_to_features(genome, net_type).reshape(1, -1)
        return float(self._model.predict(features)[0])

    def predict_batch(
        self, genomes: List[List[float]], net_type: str
    ) -> np.ndarray:
        """Predict accuracy for multiple genomes."""
        if not self._fitted:
            return np.zeros(len(genomes))
        X = np.stack([genome_to_features(g, net_type) for g in genomes])
        return self._model.predict(X)

    def uncertainty(self, genome: List[float], net_type: str) -> float:
        """Estimate prediction uncertainty (upper - mean prediction)."""
        if not self._fitted:
            return 1.0
        features = genome_to_features(genome, net_type).reshape(1, -1)
        mean_pred = float(self._model.predict(features)[0])
        upper_pred = float(self._model_upper.predict(features)[0])
        return max(upper_pred - mean_pred, 0.0)

    def uncertainty_batch(
        self, genomes: List[List[float]], net_type: str
    ) -> np.ndarray:
        """Batch uncertainty estimation."""
        if not self._fitted:
            return np.ones(len(genomes))
        X = np.stack([genome_to_features(g, net_type) for g in genomes])
        means = self._model.predict(X)
        uppers = self._model_upper.predict(X)
        return np.maximum(uppers - means, 0.0)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_observations(self) -> int:
        return len(self._X)

    @property
    def spearman_history(self) -> List[float]:
        return self._spearman_history


if __name__ == "__main__":
    from search_space.mlp_space import random_mlp_genome
    rng = np.random.default_rng(42)
    predictor = SurrogatePredictor()
    for _ in range(30):
        g = random_mlp_genome(rng)
        acc = rng.uniform(0.5, 0.99)
        predictor.add_observation(g, "mlp", acc)
    rho = predictor.fit()
    g_test = random_mlp_genome(rng)
    print(f"Predicted: {predictor.predict(g_test, 'mlp'):.4f}")
    print(f"Uncertainty: {predictor.uncertainty(g_test, 'mlp'):.4f}")
    print(f"Spearman ρ: {rho:.4f}")
