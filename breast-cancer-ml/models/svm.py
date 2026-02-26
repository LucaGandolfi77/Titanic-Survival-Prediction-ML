"""Support Vector Machine wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.svm import SVC

from models.base_model import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine classifier (SVC) with probability support.

    Args:
        params: Hyperparameters forwarded to :class:`sklearn.svm.SVC`.
    """

    name: str = "SVM"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    def _build_estimator(self) -> SVC:
        """Build an unfitted SVC estimator.

        Returns:
            Configured :class:`SVC` instance with ``probability=True``.
        """
        return SVC(
            C=self.params.get("C", 1.0),
            kernel=self.params.get("kernel", "rbf"),
            probability=self.params.get("probability", True),
            random_state=self.params.get("random_state", 42),
        )
