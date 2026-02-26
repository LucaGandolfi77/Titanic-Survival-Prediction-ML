"""Training pipeline — train all models with cross-validation.

Usage:
    python -m pipeline.train
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

# Ensure project root is on sys.path so relative imports work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_config, load_data, DataBundle  # noqa: E402
from models.logistic_regression import LogisticRegressionModel  # noqa: E402
from models.random_forest import RandomForestModel  # noqa: E402
from models.svm import SVMModel  # noqa: E402
from models.xgboost_model import XGBoostModel  # noqa: E402
from models.base_model import BaseModel  # noqa: E402


def _build_models(config: dict) -> List[BaseModel]:
    """Instantiate all model wrappers using config hyperparameters.

    Args:
        config: Parsed config.yaml.

    Returns:
        List of unfitted :class:`BaseModel` subclasses.
    """
    model_cfgs: dict = config.get("models", {})
    rs = config.get("random_state", 42)

    return [
        LogisticRegressionModel({**model_cfgs.get("logistic_regression", {}), "random_state": rs}),
        RandomForestModel({**model_cfgs.get("random_forest", {}), "random_state": rs}),
        SVMModel({**model_cfgs.get("svm", {}), "random_state": rs}),
        XGBoostModel({**model_cfgs.get("xgboost", {}), "random_state": rs}),
    ]


def cross_validate_model(
    model: BaseModel,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation on a single model.

    Args:
        model: An unfitted :class:`BaseModel`.
        X: Feature matrix.
        y: Target vector.
        cv_folds: Number of folds.
        random_state: Random seed.

    Returns:
        Dictionary with mean and std of each scoring metric.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
    }

    cv_results = cross_validate(
        model.estimator,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    metrics: Dict[str, float] = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        metrics[f"{metric_name}_mean"] = float(np.mean(cv_results[key]))
        metrics[f"{metric_name}_std"] = float(np.std(cv_results[key]))

    return metrics


def train_all(
    config: dict | None = None,
) -> Tuple[List[BaseModel], pd.DataFrame]:
    """Train every registered model and collect CV results.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (list of fitted models, DataFrame of CV results).
    """
    if config is None:
        config = load_config()

    bundle: DataBundle = load_data(config, scale=True, export_csv=True)
    models = _build_models(config)

    cv_folds: int = config.get("cv_folds", 5)
    rs: int = config.get("random_state", 42)
    models_dir = ROOT / config["paths"]["models"]
    reports_dir = ROOT / config["paths"]["reports"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_rows: list[dict] = []

    print("── Training Pipeline ─────────────────────────────")
    for model in models:
        print(f"\n▸ {model.name}")

        # Cross-validation (on training set only)
        cv_metrics = cross_validate_model(
            model, bundle.X_train, bundle.y_train, cv_folds, rs
        )
        for k, v in cv_metrics.items():
            print(f"    {k:<20s}: {v:.4f}")

        # Full train fit (on complete training set)
        model.fit(bundle.X_train, bundle.y_train)

        # Evaluate on held-out test set
        test_metrics = model.evaluate(bundle.X_test, bundle.y_test)
        print("    --- test set ---")
        for k, v in test_metrics.items():
            print(f"    {k:<20s}: {v:.4f}")

        # Save model
        slug = model.name.lower().replace(" ", "_")
        model.save(models_dir / f"{slug}.pkl")

        results_rows.append({
            "model": model.name,
            **{f"cv_{k}": v for k, v in cv_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

    # ── save CV results ──────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    csv_path = reports_dir / "cv_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ CV results → {csv_path}")
    print("── Done ──────────────────────────────────────────\n")

    return models, results_df


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_all()
