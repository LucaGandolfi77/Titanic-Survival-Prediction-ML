"""Abstract base class for all models in the breast-cancer pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseModel(ABC):
    """Abstract base class that every concrete model must inherit from.

    Subclasses must implement :meth:`_build_estimator` which returns a
    fitted or unfitted sklearn-compatible estimator.

    Attributes:
        name: Human-readable model name (e.g. ``"Logistic Regression"``).
        params: Hyperparameter dictionary drawn from *config.yaml*.
        estimator: The underlying sklearn estimator (set after fit).
    """

    name: str = "BaseModel"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = params or {}
        self.estimator: Any = self._build_estimator()

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return an **unfitted** sklearn-compatible estimator.

        Returns:
            An estimator instance with the configured hyperparameters.
        """
        ...

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "BaseModel":
        """Fit the model on the training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            ``self`` for chaining.
        """
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate class predictions.

        Args:
            X: Feature matrix.

        Returns:
            1-D array of predicted class labels.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            2-D array of shape ``(n_samples, n_classes)``.
        """
        return self.estimator.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """Compute a dictionary of evaluation metrics.

        Args:
            X_test: Test feature matrix.
            y_test: True labels.

        Returns:
            Dictionary with accuracy, precision, recall, f1, and roc_auc.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

    def save(self, path: Path) -> None:
        """Serialise the fitted estimator to disk.

        Args:
            path: File path (typically ``*.pkl``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        print(f"  ✓ Saved model → {path}")

    @classmethod
    def load(cls, path: Path) -> Any:
        """Deserialise a previously saved estimator.

        Args:
            path: Path to the ``.pkl`` file.

        Returns:
            The loaded estimator object.
        """
        return joblib.load(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
