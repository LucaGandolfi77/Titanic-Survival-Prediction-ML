"""
optuna_optimizer.py — Hyperparameter Optimization with Optuna + MLflow
======================================================================
Runs Bayesian hyperparameter search via Optuna, logging every trial
to MLflow so the full experiment history is traceable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.models.evaluator import evaluate_model
from src.utils.mlflow_utils import init_mlflow

logger = logging.getLogger("titanic_mlops.optuna_optimizer")


def _build_xgb_params(trial: optuna.Trial, search_space: Dict) -> Dict[str, Any]:
    """
    Sample XGBoost hyperparameters from the Optuna search space.

    Parameters
    ----------
    trial : optuna.Trial
    search_space : dict
        Ranges defined in config.yaml.

    Returns
    -------
    dict
        Sampled hyperparameters.
    """
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            search_space["n_estimators"][0],
            search_space["n_estimators"][1],
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            search_space["max_depth"][0],
            search_space["max_depth"][1],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            search_space["learning_rate"][0],
            search_space["learning_rate"][1],
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            search_space["subsample"][0],
            search_space["subsample"][1],
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            search_space["colsample_bytree"][0],
            search_space["colsample_bytree"][1],
        ),
    }
    return params


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
):
    """
    Factory that returns an Optuna objective function closed over
    the training data and configuration.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
    cfg : dict

    Returns
    -------
    callable
        The objective function for ``study.optimize()``.
    """
    seed = cfg["project"]["random_seed"]
    cv_folds = cfg["training"]["cv_folds"]
    search_space = cfg["optuna"]["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = _build_xgb_params(trial, search_space)

        model = XGBClassifier(
            **params,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        # Cross-validation score as the optimisation target
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring="accuracy",
        )
        cv_mean = float(cv_scores.mean())

        # Also evaluate on hold-out for monitoring (not used for selection)
        model.fit(X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)

        # Log everything to MLflow nested run
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_accuracy", cv_mean)
            mlflow.log_metrics(val_metrics)
            mlflow.set_tag("optuna_trial", trial.number)

        return cv_mean

    return objective


def run_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the full Optuna study and return the best model + params.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
    cfg : dict

    Returns
    -------
    dict
        ``{"best_params": {...}, "best_score": float, "model": fitted_model, "study": Study}``
    """
    optuna_cfg = cfg["optuna"]
    seed = cfg["project"]["random_seed"]

    init_mlflow(cfg)

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name="xgboost-hpo",
        direction="maximize",
        sampler=sampler,
    )

    objective = create_objective(X_train, y_train, X_val, y_val, cfg)

    # Parent MLflow run wrapping all trials
    with mlflow.start_run(run_name="optuna-hpo") as parent_run:
        mlflow.set_tag("stage", "hpo")

        study.optimize(
            objective,
            n_trials=optuna_cfg["n_trials"],
            timeout=optuna_cfg.get("timeout"),
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_score = study.best_value

        logger.info("Best trial: %s  →  cv_accuracy=%.4f", best_params, best_score)

        # Retrain best model on full training data
        best_model = XGBClassifier(
            **best_params,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        best_model.fit(X_train, y_train)

        # Final validation metrics
        val_metrics = evaluate_model(best_model, X_val, y_val)

        mlflow.log_params(best_params)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metric("best_cv_accuracy", best_score)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name=cfg["mlflow"]["registry_name"],
        )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "model": best_model,
        "metrics": val_metrics,
        "study": study,
        "parent_run_id": parent_run.info.run_id,
    }
