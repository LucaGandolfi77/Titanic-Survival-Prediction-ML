"""
registry.py – Central model catalog.

Every supported algorithm is registered here with its scikit-learn-compatible
constructor and metadata.  The YAML config provides default hyper-parameter
ranges; this module maps string keys → actual estimator classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from sklearn.base import BaseEstimator

# ── Sklearn models ────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── Boosting libraries ────────────────────────────────────────
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"


@dataclass
class ModelInfo:
    """Metadata wrapper for a model."""
    key: str
    display_name: str
    task: str  # "classification" | "regression"
    estimator_class: Type[BaseEstimator]
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_spec: Dict[str, Any] = field(default_factory=dict)


# ── Mapping from YAML key → estimator class ──────────────────

_CLASS_MAP: Dict[str, Type[BaseEstimator]] = {
    # Classification
    "logistic_regression": LogisticRegression,
    "random_forest_clf": RandomForestClassifier,
    "gradient_boosting_clf": GradientBoostingClassifier,
    "svm_clf": SVC,
    "knn_clf": KNeighborsClassifier,
    "xgboost_clf": XGBClassifier,
    "lightgbm_clf": LGBMClassifier,
    "catboost_clf": CatBoostClassifier,
    # Regression
    "linear_regression": LinearRegression,
    "ridge_regression": Ridge,
    "random_forest_reg": RandomForestRegressor,
    "gradient_boosting_reg": GradientBoostingRegressor,
    "xgboost_reg": XGBRegressor,
    "lightgbm_reg": LGBMRegressor,
    "catboost_reg": CatBoostRegressor,
}


def _load_config() -> dict:
    """Read the YAML config file."""
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def _build_registry() -> Dict[str, ModelInfo]:
    """Build the full model registry from YAML + class map."""
    cfg = _load_config()
    registry: Dict[str, ModelInfo] = {}

    for key, cls in _CLASS_MAP.items():
        section = cfg.get(key, {})
        params_spec = section.get("params", {})
        # Extract defaults from spec
        defaults = {
            p: spec["default"]
            for p, spec in params_spec.items()
            if "default" in spec
        }

        # Inject quiet-mode defaults for boosting libs
        if cls in (CatBoostClassifier, CatBoostRegressor):
            defaults.setdefault("verbose", 0)
        if cls in (LGBMClassifier, LGBMRegressor):
            defaults.setdefault("verbose", -1)
        if cls in (XGBClassifier, XGBRegressor):
            defaults.setdefault("verbosity", 0)
            defaults.setdefault("use_label_encoder", False)

        registry[key] = ModelInfo(
            key=key,
            display_name=section.get("display_name", key),
            task=section.get("task", "classification"),
            estimator_class=cls,
            default_params=defaults,
            param_spec=params_spec,
        )

    return registry


# Singleton registry
MODEL_REGISTRY: Dict[str, ModelInfo] = _build_registry()


# ── Public helpers ────────────────────────────────────────────

def get_models_for_task(task: str) -> List[ModelInfo]:
    """Return all registered models for the given task ('classification' | 'regression')."""
    return [m for m in MODEL_REGISTRY.values() if m.task == task]


def get_model(key: str) -> ModelInfo:
    """Retrieve a single model by its registry key."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    return MODEL_REGISTRY[key]


def create_estimator(key: str, params: Dict[str, Any] | None = None) -> BaseEstimator:
    """Instantiate an estimator with the given (or default) hyper-parameters."""
    info = get_model(key)
    merged = {**info.default_params, **(params or {})}

    # SVC needs probability=True for ROC curves
    if info.estimator_class is SVC:
        merged.setdefault("probability", True)

    return info.estimator_class(**merged)
