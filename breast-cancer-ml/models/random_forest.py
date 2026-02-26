"""Random Forest wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier.

    Args:
        params: Hyperparameters forwarded to
            :class:`sklearn.ensemble.RandomForestClassifier`.
    """

    name: str = "Random Forest"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    def _build_estimator(self) -> RandomForestClassifier:
        """Build an unfitted RandomForestClassifier estimator.

        Returns:
            Configured :class:`RandomForestClassifier` instance.
        """
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", None),
            min_samples_split=self.params.get("min_samples_split", 2),
            random_state=self.params.get("random_state", 42),
        )
