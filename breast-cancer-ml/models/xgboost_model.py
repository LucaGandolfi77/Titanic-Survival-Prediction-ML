"""XGBoost wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from xgboost import XGBClassifier

from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost gradient-boosted tree classifier.

    Args:
        params: Hyperparameters forwarded to
            :class:`xgboost.XGBClassifier`.
    """

    name: str = "XGBoost"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    def _build_estimator(self) -> XGBClassifier:
        """Build an unfitted XGBClassifier estimator.

        Returns:
            Configured :class:`XGBClassifier` instance.
        """
        return XGBClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            learning_rate=self.params.get("learning_rate", 0.1),
            max_depth=self.params.get("max_depth", 5),
            use_label_encoder=self.params.get("use_label_encoder", False),
            eval_metric=self.params.get("eval_metric", "logloss"),
            random_state=self.params.get("random_state", 42),
            verbosity=0,
        )
