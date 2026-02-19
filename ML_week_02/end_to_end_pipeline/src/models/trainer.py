"""
trainer.py — Training Orchestration
====================================
Trains multiple classifiers defined in config.yaml, evaluates
each with cross-validation, and logs everything to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.models.evaluator import evaluate_model
from src.utils.mlflow_utils import init_mlflow

logger = logging.getLogger("titanic_mlops.trainer")

# ── Model factory ────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def _build_model(name: str, params: Dict[str, Any], seed: int) -> Any:
    """Instantiate a model by name with given hyperparameters."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")

    # Inject reproducibility
    init_kwargs = {**params}
    if name == "xgboost":
        init_kwargs.setdefault("random_state", seed)
        init_kwargs.setdefault("use_label_encoder", False)
        init_kwargs.setdefault("eval_metric", "logloss")
        init_kwargs.setdefault("verbosity", 0)
    elif hasattr(cls, "random_state"):
        init_kwargs.setdefault("random_state", seed)

    return cls(**init_kwargs)


# ── Training loop ────────────────────────────────────────────


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Train every model listed in config, log to MLflow, and return results.

    Parameters
    ----------
    X_train, y_train : ndarray
        Processed training data.
    X_val, y_val : ndarray
        Processed validation data.
    cfg : dict
        Full project configuration.

    Returns
    -------
    results : dict
        ``{model_name: {"model": fitted_model, "metrics": {...}, "run_id": str}}``
    """
    seed = cfg["project"]["random_seed"]
    training_cfg = cfg["training"]
    cv_folds = training_cfg["cv_folds"]
    models_dir: Path = cfg["paths"]["models_dir"]
    registry_name = cfg["mlflow"]["registry_name"]

    init_mlflow(cfg)

    results: Dict[str, Dict[str, Any]] = {}

    for model_name, model_params in training_cfg["models"].items():
        logger.info("━━━ Training: %s ━━━", model_name)

        model = _build_model(model_name, model_params, seed)

        # Fit
        model.fit(X_train, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring="accuracy",
        )

        # Full evaluation on hold-out validation set
        metrics = evaluate_model(model, X_val, y_val)
        metrics["cv_mean_accuracy"] = float(cv_scores.mean())
        metrics["cv_std_accuracy"] = float(cv_scores.std())

        # Log to MLflow
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "baseline")
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=registry_name,
            )
            run_id = run.info.run_id

        # Save locally as well
        local_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, local_path)
        logger.info("Saved model locally → %s", local_path)

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "run_id": run_id,
        }

        logger.info(
            "%s  →  val_accuracy=%.4f  cv_accuracy=%.4f±%.4f",
            model_name,
            metrics["accuracy"],
            metrics["cv_mean_accuracy"],
            metrics["cv_std_accuracy"],
        )

    return results


def train_single_model(
    model_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    """
    Train a single model (used by the Optuna optimizer).

    Parameters
    ----------
    model_name : str
        Key from MODEL_REGISTRY.
    params : dict
        Hyperparameters.
    X_train, y_train : ndarray
        Training data.
    seed : int
        Random seed.

    Returns
    -------
    Fitted model.
    """
    model = _build_model(model_name, params, seed)
    model.fit(X_train, y_train)
    return model
