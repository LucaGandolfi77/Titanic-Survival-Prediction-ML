"""
lime_explainer.py – LIME-based local explanations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import BaseEstimator


@dataclass
class LIMEResult:
    """Container for a single LIME explanation."""
    feature_weights: pd.DataFrame  # feature, weight, value
    prediction: float | int
    prediction_proba: Optional[np.ndarray]
    intercept: float
    score: float  # local fidelity R²
    raw_explanation: Any  # lime Explanation object


class LIMEExplainer:
    """Re-usable LIME explainer for a fixed model + training data."""

    def __init__(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        class_names: list[str] | None = None,
        mode: str = "classification",
        n_samples: int = 5000,
        n_features: int = 10,
    ):
        self.estimator = estimator
        self.mode = mode
        self.n_features = n_features

        feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else None

        # Detect categorical columns
        cat_indices = []
        if hasattr(X_train, "dtypes"):
            for i, dt in enumerate(X_train.dtypes):
                if dt.kind in ("O", "b") or str(dt) == "category":
                    cat_indices.append(i)

        self._explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            categorical_features=cat_indices or None,
            discretize_continuous=True,
            random_state=42,
            sample_around_instance=True,
        )
        self._n_samples = n_samples

    def explain_instance(
        self,
        row: pd.DataFrame | pd.Series | np.ndarray,
        n_features: int | None = None,
    ) -> LIMEResult:
        """Generate a LIME explanation for a single observation."""
        if isinstance(row, pd.DataFrame):
            row_arr = row.values.flatten()
        elif isinstance(row, pd.Series):
            row_arr = row.values
        else:
            row_arr = row.flatten()

        predict_fn = (
            self.estimator.predict_proba
            if hasattr(self.estimator, "predict_proba") and self.mode == "classification"
            else self.estimator.predict
        )

        n_feat = n_features or self.n_features

        exp = self._explainer.explain_instance(
            row_arr,
            predict_fn,
            num_features=n_feat,
            num_samples=self._n_samples,
        )

        # Parse feature weights
        weights = []
        for feat_desc, weight in exp.as_list():
            weights.append({"feature_rule": feat_desc, "weight": round(weight, 6)})

        weights_df = pd.DataFrame(weights).sort_values("weight", key=abs, ascending=False).reset_index(drop=True)

        # Prediction
        pred = self.estimator.predict(row_arr.reshape(1, -1))[0]
        proba = None
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(row_arr.reshape(1, -1))[0]

        # Safely extract intercept — LIME stores it as a dict for classification
        _intercept = exp.intercept
        if isinstance(_intercept, dict):
            # Pick class-1 intercept if available, else first value
            _intercept = _intercept.get(1, next(iter(_intercept.values())))
        elif hasattr(_intercept, "__len__") and len(_intercept) > 1:
            _intercept = float(_intercept[1])
        elif hasattr(_intercept, "__len__"):
            _intercept = float(_intercept[0])
        else:
            _intercept = float(_intercept)

        return LIMEResult(
            feature_weights=weights_df,
            prediction=pred,
            prediction_proba=proba,
            intercept=_intercept,
            score=round(exp.score, 4),
            raw_explanation=exp,
        )
