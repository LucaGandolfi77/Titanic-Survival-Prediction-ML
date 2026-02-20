"""
Model registry / catalog.

Maps string names to sklearn-compatible estimator factories + their
Optuna search spaces.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.svm import SVC, SVR

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _safe_import(module: str, cls: str):
    """Import a class, returning None if unavailable."""
    try:
        m = __import__(module, fromlist=[cls])
        return getattr(m, cls)
    except (ImportError, AttributeError):
        return None


XGBClassifier = _safe_import("xgboost", "XGBClassifier")
XGBRegressor = _safe_import("xgboost", "XGBRegressor")
LGBMClassifier = _safe_import("lightgbm", "LGBMClassifier")
LGBMRegressor = _safe_import("lightgbm", "LGBMRegressor")
CatBoostClassifier = _safe_import("catboost", "CatBoostClassifier")
CatBoostRegressor = _safe_import("catboost", "CatBoostRegressor")


# ── Registry ─────────────────────────────────────────────────────


class ModelEntry:
    """Single entry in the model registry."""

    def __init__(
        self,
        name: str,
        clf_class: type | None,
        reg_class: type | None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.clf_class = clf_class
        self.reg_class = reg_class
        self.default_params = default_params or {}

    def build(self, task: str, **overrides) -> Any:
        cls = self.clf_class if task == "classification" else self.reg_class
        if cls is None:
            raise ImportError(f"No {task} class for model '{self.name}'")
        params = {**self.default_params, **overrides}
        # Filter params to only those accepted by the target class
        sig = inspect.signature(cls)
        valid_keys = set(sig.parameters.keys())
        # If the constructor accepts **kwargs, pass everything through
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_keyword:
            dropped = set(params) - valid_keys
            if dropped:
                logger.debug(
                    f"Dropping params {dropped} not accepted by {cls.__name__}"
                )
            params = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**params)


_REGISTRY: dict[str, ModelEntry] = {
    "logistic_regression": ModelEntry(
        "logistic_regression",
        LogisticRegression,
        Ridge,
        {"max_iter": 1000},
    ),
    "random_forest": ModelEntry(
        "random_forest",
        RandomForestClassifier,
        RandomForestRegressor,
        {"n_jobs": -1, "random_state": 42},
    ),
    "extra_trees": ModelEntry(
        "extra_trees",
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        {"n_jobs": -1, "random_state": 42},
    ),
    "xgboost": ModelEntry(
        "xgboost",
        XGBClassifier,
        XGBRegressor,
        {
            "verbosity": 0,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
        },
    ),
    "lightgbm": ModelEntry(
        "lightgbm",
        LGBMClassifier,
        LGBMRegressor,
        {
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "catboost": ModelEntry(
        "catboost",
        CatBoostClassifier,
        CatBoostRegressor,
        {
            "verbose": 0,
            "random_state": 42,
            "thread_count": -1,
        },
    ),
    "svm": ModelEntry(
        "svm",
        SVC,
        SVR,
        {"probability": True, "kernel": "rbf"},
    ),
}


def get_model_entry(name: str) -> ModelEntry:
    entry = _REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown model: {name}. Available: {list(_REGISTRY)}")
    return entry


def list_available_models() -> list[str]:
    """Return names of models whose libraries are actually installed."""
    available = []
    for name, entry in _REGISTRY.items():
        if entry.clf_class is not None or entry.reg_class is not None:
            available.append(name)
    return available
