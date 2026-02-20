"""
Bayesian hyper-parameter optimisation  (Stage 4).

Uses Optuna TPE sampler + MedianPruner.
Each trial:
  1. Sample hyper-parameters from config-defined search space.
  2. Run k-fold CV.
  3. Report score to Optuna (with MLflow nested run).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ..utils.logger import get_logger
from .registry import get_model_entry

logger = get_logger(__name__)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None  # type: ignore

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore


class BayesianOptimizer:
    """Optuna-based Bayesian HPO for each top-K model."""

    def __init__(self, config) -> None:
        self.n_trials: int = getattr(config, "n_trials", 100)
        self.cv_folds: int = getattr(config, "cv_folds", 5)
        self.timeout: Optional[int] = getattr(config, "timeout_seconds", None)
        self.search_spaces: dict = getattr(config, "search_spaces", {})
        self.enable_mlflow: bool = getattr(config, "enable_mlflow", False)
        self.best_params_: Dict[str, Dict[str, Any]] = {}
        self.best_scores_: Dict[str, float] = {}
        self.studies_: Dict[str, Any] = {}

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        model_names: List[str],
        task: str,
    ) -> Dict[str, Any]:
        """Run Optuna for each model. Returns dict of {name: fitted_best_model}."""
        if optuna is None:
            raise ImportError("pip install optuna")

        scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
        direction = "maximize"
        best_models: Dict[str, Any] = {}

        for name in model_names:
            logger.info(f"  Optimising {name} ({self.n_trials} trials) …")
            space_cfg = self._get_space(name, task)

            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )

            def objective(trial: optuna.Trial, _name=name, _space=space_cfg) -> float:
                params = self._sample(trial, _space)
                entry = get_model_entry(_name)
                model = entry.build(task, **params)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        model, X, y,
                        cv=self.cv_folds,
                        scoring=scoring,
                        n_jobs=-1,
                        error_score=float("-inf"),
                    )
                mean_score = float(scores.mean())

                # Log to MLflow if enabled
                if self.enable_mlflow and mlflow is not None:
                    try:
                        with mlflow.start_run(nested=True, run_name=f"{_name}_trial_{trial.number}"):
                            mlflow.log_params(params)
                            mlflow.log_metric("cv_score", mean_score)
                    except Exception:
                        pass

                return mean_score

            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
                show_progress_bar=False,
            )

            self.studies_[name] = study
            self.best_params_[name] = study.best_params
            self.best_scores_[name] = study.best_value

            # Refit best model on full data
            entry = get_model_entry(name)
            best_model = entry.build(task, **study.best_params)
            best_model.fit(X, y)
            best_models[name] = best_model

            logger.info(
                f"  {name} best score={study.best_value:.4f} "
                f"params={study.best_params}"
            )

        return best_models

    # ── Search space handling ──────────────────────────────────────

    def _get_space(self, name: str, task: str) -> dict:
        """Return search space dict from config or a sensible default.

        For models with task-specific default spaces (e.g.
        logistic_regression → LogisticRegression / Ridge), always use
        the built-in task-aware space since the config-provided space
        was written for one variant and may contain incompatible
        param names or value ranges for the other.
        """
        default = _DEFAULT_SPACES.get(name, {})
        is_task_specific = "classification" in default or "regression" in default

        # Models with separate clf/reg classes need task-specific spaces
        if is_task_specific:
            return default.get(task, {})

        # Otherwise, prefer config-provided space
        cfg_space = getattr(self.search_spaces, name, None)
        if cfg_space is not None:
            return dict(cfg_space) if hasattr(cfg_space, "__iter__") else {}

        return default

    @staticmethod
    def _sample(trial: optuna.Trial, space: dict) -> dict:
        """Convert config search-space entries to Optuna suggestions."""
        params: Dict[str, Any] = {}
        for key, spec in space.items():
            if isinstance(spec, dict):
                # Structured spec: {"type": "int", "low": 50, "high": 500, "log": true}
                stype = spec.get("type", "float")
                if stype == "int":
                    params[key] = trial.suggest_int(
                        key, spec["low"], spec["high"],
                        log=spec.get("log", False),
                    )
                elif stype == "float":
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"],
                        log=spec.get("log", False),
                    )
                elif stype == "categorical":
                    params[key] = trial.suggest_categorical(key, spec["choices"])
                else:
                    params[key] = trial.suggest_float(key, spec.get("low", 0), spec.get("high", 1))
            elif isinstance(spec, list):
                params[key] = trial.suggest_categorical(key, spec)
            else:
                params[key] = spec  # fixed value
        return params


# ── Fallback built-in search spaces ──────────────────────────────

_DEFAULT_SPACES: Dict[str, Dict[str, Any]] = {
    # Task-specific: LogisticRegression (classification) vs Ridge (regression)
    "logistic_regression": {
        "classification": {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
            "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
        },
        "regression": {
            "alpha": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
        },
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    },
    "extra_trees": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 20, "high": 150},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "catboost": {
        "iterations": {"type": "int", "low": 50, "high": 500},
        "depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
    },
    # Task-specific: SVC (classification) vs SVR (regression)
    "svm": {
        "classification": {
            "C": {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        },
        "regression": {
            "C": {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            "epsilon": {"type": "float", "low": 0.01, "high": 1.0},
        },
    },
}
