"""
shap_explainer.py – SHAP-based global and local explanations.

Provides a high-level wrapper that auto-selects the right SHAP algorithm
(TreeExplainer, LinearExplainer, KernelExplainer) depending on the model type.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


# ── Algorithm auto-selection ──────────────────────────────────

_TREE_CLASSES = (
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
)

_LINEAR_CLASSES = (
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
    "SGDClassifier", "SGDRegressor",
)


def _select_explainer(estimator: BaseEstimator, X_background: pd.DataFrame | np.ndarray):
    """Auto-pick the best SHAP algorithm."""
    cls_name = type(estimator).__name__

    if cls_name in _TREE_CLASSES:
        return shap.TreeExplainer(estimator)

    if cls_name in _LINEAR_CLASSES:
        return shap.LinearExplainer(estimator, X_background)

    # Fallback: KernelExplainer (model-agnostic, slower)
    bg = shap.sample(X_background, min(100, len(X_background)))
    predict_fn = (
        estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    )
    return shap.KernelExplainer(predict_fn, bg)


# ── Public API ────────────────────────────────────────────────

class SHAPExplainer:
    """Cached SHAP explanation engine for one model + dataset pair."""

    def __init__(
        self,
        estimator: BaseEstimator,
        X_background: pd.DataFrame,
        algorithm: str = "auto",
        check_additivity: bool = False,
    ):
        self.estimator = estimator
        self.X_background = X_background
        self.check_additivity = check_additivity

        if algorithm == "auto":
            self._explainer = _select_explainer(estimator, X_background)
        elif algorithm == "tree":
            self._explainer = shap.TreeExplainer(estimator)
        elif algorithm == "linear":
            self._explainer = shap.LinearExplainer(estimator, X_background)
        elif algorithm == "kernel":
            bg = shap.sample(X_background, min(100, len(X_background)))
            pred_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
            self._explainer = shap.KernelExplainer(pred_fn, bg)
        else:
            raise ValueError(f"Unknown SHAP algorithm: {algorithm}")

        self._shap_values: Optional[np.ndarray] = None
        self._expected_value = None

    # ── Global explanations ───────────────────────────────────

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> np.ndarray:
        """Compute SHAP values for X (sub-sampled if needed)."""
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)

        sv = self._explainer.shap_values(X, check_additivity=self.check_additivity)
        self._shap_values = sv
        self._expected_value = self._explainer.expected_value
        return sv

    def global_importance(self, X: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
        """Mean |SHAP| per feature → sorted importance table."""
        sv = self.compute_shap_values(X, max_samples)

        # Handle multi-class (list of arrays)
        if isinstance(sv, list):
            sv = np.abs(np.array(sv)).mean(axis=0)

        mean_abs = np.abs(sv).mean(axis=0)
        # If still multi-dimensional (binary classification gives shape (n, features, 2))
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1)

        feature_names = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]

        df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs.flatten(),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    @property
    def expected_value(self):
        return self._expected_value

    @property
    def shap_values(self):
        return self._shap_values

    # ── Local explanations ────────────────────────────────────

    def explain_instance(self, row: pd.DataFrame | pd.Series) -> Dict[str, Any]:
        """SHAP explanation for a single data point."""
        if isinstance(row, pd.Series):
            row = row.to_frame().T

        sv = self._explainer.shap_values(row, check_additivity=self.check_additivity)

        # For binary classification keep positive-class SHAP
        if isinstance(sv, list):
            sv_single = sv[1][0] if len(sv) == 2 else sv[0][0]
            base = self._explainer.expected_value[1] if isinstance(self._explainer.expected_value, (list, np.ndarray)) else self._explainer.expected_value
        else:
            sv_single = sv[0]
            # Handle multi-dimensional: shape (1, features, classes) → take class 1
            if sv_single.ndim > 1:
                sv_single = sv_single[:, 1] if sv_single.shape[1] > 1 else sv_single[:, 0]
            base = self._explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[1] if len(base) > 1 else base[0]

        feature_names = row.columns.tolist() if hasattr(row, "columns") else [f"f{i}" for i in range(row.shape[1])]

        sv_flat = np.asarray(sv_single).flatten()

        contributions = pd.DataFrame({
            "feature": feature_names,
            "value": row.values.flatten(),
            "shap_value": sv_flat,
        }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

        return {
            "base_value": float(base) if np.isscalar(base) else float(np.array(base).item()),
            "contributions": contributions,
            "prediction_offset": float(sv_flat.sum()),
        }

    # ── Feature interaction ───────────────────────────────────

    def interaction_values(self, X: pd.DataFrame, max_samples: int = 200) -> Optional[np.ndarray]:
        """SHAP interaction values (TreeExplainer only)."""
        if not isinstance(self._explainer, shap.TreeExplainer):
            return None
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        return self._explainer.shap_interaction_values(X)
