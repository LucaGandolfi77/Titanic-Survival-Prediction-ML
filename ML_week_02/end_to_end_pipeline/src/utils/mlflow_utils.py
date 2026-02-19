"""
mlflow_utils.py — MLflow Helper Functions
==========================================
Centralises MLflow setup, experiment creation, and model-registry
operations so that callers never interact with the raw MLflow API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger("titanic_mlops.mlflow_utils")


def init_mlflow(cfg: Dict[str, Any]) -> str:
    """
    Initialise MLflow tracking and return the experiment ID.

    Parameters
    ----------
    cfg : dict
        Full project configuration (as returned by config_loader.load_config).

    Returns
    -------
    str
        The MLflow experiment ID.
    """
    mlflow_cfg = cfg["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI set to %s", tracking_uri)

    experiment = mlflow.get_experiment_by_name(mlflow_cfg["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            mlflow_cfg["experiment_name"],
            artifact_location=mlflow_cfg.get("artifact_location"),
        )
        logger.info("Created MLflow experiment: %s (id=%s)",
                     mlflow_cfg["experiment_name"], experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using existing MLflow experiment: %s (id=%s)",
                     mlflow_cfg["experiment_name"], experiment_id)

    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    return experiment_id


def log_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, Path]] = None,
    tags: Optional[Dict[str, str]] = None,
    model: Any = None,
    model_name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> str:
    """
    Log a complete training run to MLflow.

    Parameters
    ----------
    params : dict
        Hyperparameters.
    metrics : dict
        Evaluation metrics.
    artifacts : dict, optional
        Mapping of artifact_name → local file path.
    tags : dict, optional
        Run-level tags.
    model : sklearn estimator, optional
        Trained model to log as an MLflow artifact.
    model_name : str, optional
        Artifact path name inside the run (default "model").
    registered_model_name : str, optional
        If provided, also register the model in the Model Registry.

    Returns
    -------
    str
        The MLflow run ID.
    """
    with mlflow.start_run() as run:
        # Parameters
        mlflow.log_params(params)

        # Metrics
        mlflow.log_metrics(metrics)

        # Tags
        if tags:
            mlflow.set_tags(tags)

        # Artifacts (extra files)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(str(path), artifact_path=name)

        # Sklearn model
        if model is not None:
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name or "model",
                registered_model_name=registered_model_name,
            )

        run_id = run.info.run_id
        logger.info("Logged MLflow run %s", run_id)
        return run_id


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str = "Production",
    archive_existing: bool = True,
) -> None:
    """
    Transition a registered model version to a new stage.

    Parameters
    ----------
    model_name : str
        Registered model name in the Model Registry.
    version : int
        Model version number.
    stage : str
        Target stage (Staging | Production | Archived | None).
    archive_existing : bool
        Whether to archive the current model in the target stage.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("Model %s v%d → stage '%s'", model_name, version, stage)


def load_production_model(model_name: str, stage: str = "Production") -> Any:
    """
    Load the latest model from a given stage in the Model Registry.

    Parameters
    ----------
    model_name : str
        Registered model name.
    stage : str
        Stage to load from.

    Returns
    -------
    sklearn estimator
        The loaded model.
    """
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)
