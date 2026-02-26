"""Tests for training, evaluation and prediction pipeline modules."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import DataBundle, load_config, load_data
from models.logistic_regression import LogisticRegressionModel
from pipeline.train import _build_models, cross_validate_model, train_all
from pipeline.predict import predict, load_model


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def bundle(config: dict) -> DataBundle:
    return load_data(config, scale=True, export_csv=False)


# ── Tests: train ─────────────────────────────────────────────────
class TestTrainPipeline:
    """Tests for pipeline.train module."""

    def test_build_models_returns_four(self, config: dict) -> None:
        models = _build_models(config)
        assert len(models) == 4
        names = {m.name for m in models}
        assert "Logistic Regression" in names
        assert "Random Forest" in names
        assert "SVM" in names
        assert "XGBoost" in names

    def test_cross_validate_returns_metrics(
        self, config: dict, bundle: DataBundle
    ) -> None:
        model = LogisticRegressionModel({"random_state": 42})
        metrics = cross_validate_model(
            model, bundle.X_train, bundle.y_train, cv_folds=3, random_state=42
        )
        assert isinstance(metrics, dict)
        assert "accuracy_mean" in metrics
        assert "f1_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0

    def test_train_all_produces_results(self, config: dict) -> None:
        models, results_df = train_all(config)
        assert len(models) == 4
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 4
        assert "model" in results_df.columns


# ── Tests: predict ───────────────────────────────────────────────
class TestPredictPipeline:
    """Tests for pipeline.predict module."""

    def test_predict_returns_dict(self, config: dict) -> None:
        """Predict using a trained model (requires train_all to have run)."""
        # Train first so model files exist
        train_all(config)

        # First sample from the dataset
        from sklearn.datasets import load_breast_cancer
        X, _ = load_breast_cancer(return_X_y=True)
        sample = X[0].tolist()

        result = predict("logistic_regression", sample, config)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "class_label" in result
        assert result["prediction"] in (0, 1)

    def test_predict_known_malignant(self, config: dict) -> None:
        """First sample in the dataset is malignant (class 0)."""
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)

        # Use a known malignant sample
        malignant_idx = np.where(y == 0)[0][0]
        sample = X[malignant_idx].tolist()

        result = predict("logistic_regression", sample, config)
        # Should predict malignant
        assert result["prediction"] == 0
        assert result["class_label"] == "malignant"

    def test_predict_known_benign(self, config: dict) -> None:
        """Use a known benign sample."""
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)

        benign_idx = np.where(y == 1)[0][0]
        sample = X[benign_idx].tolist()

        result = predict("logistic_regression", sample, config)
        assert result["prediction"] == 1
        assert result["class_label"] == "benign"

    def test_load_model_not_found(self, config: dict) -> None:
        """Loading a non-existent model should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model", config)
