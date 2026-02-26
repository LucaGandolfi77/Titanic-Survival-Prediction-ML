"""Logistic Regression wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier.

    Args:
        params: Hyperparameters forwarded to
            :class:`sklearn.linear_model.LogisticRegression`.
    """

    name: str = "Logistic Regression"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    def _build_estimator(self) -> LogisticRegression:
        """Build an unfitted LogisticRegression estimator.

        Returns:
            Configured :class:`LogisticRegression` instance.
        """
        return LogisticRegression(
            C=self.params.get("C", 1.0),
            max_iter=self.params.get("max_iter", 1000),
            solver=self.params.get("solver", "lbfgs"),
            random_state=self.params.get("random_state", 42),
        )
