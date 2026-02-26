"""Tests for core/ml.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ml import (
    ALL_ALGORITHMS,
    CLASSIFICATION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
    MLLab,
    REGRESSION_ALGORITHMS,
    TrainResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    return pd.DataFrame({
        "x1": x,
        "x2": np.random.randn(n),
        "y": 3 * x + np.random.randn(n) * 0.5,
    })


@pytest.fixture
def clustering_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "a": np.concatenate([np.random.randn(20) - 3, np.random.randn(20), np.random.randn(20) + 3]),
        "b": np.concatenate([np.random.randn(20) - 3, np.random.randn(20), np.random.randn(20) + 3]),
    })


@pytest.fixture
def lab() -> MLLab:
    return MLLab()


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

class TestAlgorithmRegistry:
    def test_classification_has_entries(self) -> None:
        assert len(CLASSIFICATION_ALGORITHMS) >= 4

    def test_regression_has_entries(self) -> None:
        assert len(REGRESSION_ALGORITHMS) >= 4

    def test_clustering_has_entries(self) -> None:
        assert len(CLUSTERING_ALGORITHMS) >= 2

    def test_all_algorithms_keys(self) -> None:
        assert set(ALL_ALGORITHMS.keys()) == {"classification", "regression", "clustering"}

    def test_each_algo_has_required_keys(self) -> None:
        for task, registry in ALL_ALGORITHMS.items():
            for name, info in registry.items():
                assert "class" in info, f"{task}/{name} missing 'class'"
                assert "defaults" in info, f"{task}/{name} missing 'defaults'"
                assert "params" in info, f"{task}/{name} missing 'params'"


# ---------------------------------------------------------------------------
# Classification training
# ---------------------------------------------------------------------------

class TestClassificationTraining:
    @pytest.mark.parametrize("algo", ["Logistic Regression", "Random Forest", "KNN"])
    def test_train_returns_result(self, lab: MLLab, classification_df: pd.DataFrame, algo: str) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2", "f3"],
            "classification", algo, {}, 0.3,
        )
        assert isinstance(result, TrainResult)
        assert result.task == "classification"
        assert result.algorithm == algo
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert 0 <= result.metrics["accuracy"] <= 1

    def test_svm_classification(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2"],
            "classification", "SVM", {"C": "1.0", "kernel": "rbf"}, 0.3,
        )
        assert "accuracy" in result.metrics

    def test_confusion_matrix_present(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2", "f3"],
            "classification", "Random Forest", {}, 0.3,
        )
        assert result.confusion_mat is not None
        assert result.confusion_mat.shape[0] == 2  # binary

    def test_feature_importances(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2", "f3"],
            "classification", "Random Forest", {}, 0.3,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 3

    def test_string_target_encoded(self, lab: MLLab) -> None:
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "label": np.random.choice(["cat", "dog"], 50),
        })
        result = lab.train(df, "label", ["x"], "classification", "Logistic Regression", {}, 0.3)
        assert "accuracy" in result.metrics


# ---------------------------------------------------------------------------
# Regression training
# ---------------------------------------------------------------------------

class TestRegressionTraining:
    @pytest.mark.parametrize("algo", ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor"])
    def test_train_returns_result(self, lab: MLLab, regression_df: pd.DataFrame, algo: str) -> None:
        result = lab.train(
            regression_df, "y", ["x1", "x2"],
            "regression", algo, {}, 0.3,
        )
        assert isinstance(result, TrainResult)
        assert "mae" in result.metrics
        assert "mse" in result.metrics
        assert "rmse" in result.metrics
        assert "r2" in result.metrics

    def test_r2_reasonable(self, lab: MLLab, regression_df: pd.DataFrame) -> None:
        result = lab.train(
            regression_df, "y", ["x1", "x2"],
            "regression", "Linear Regression", {}, 0.2,
        )
        # y = 3*x1 + noise, so R² should be decent
        assert result.metrics["r2"] > 0.5

    def test_y_test_y_pred_present(self, lab: MLLab, regression_df: pd.DataFrame) -> None:
        result = lab.train(
            regression_df, "y", ["x1", "x2"],
            "regression", "Ridge", {}, 0.3,
        )
        assert result.y_test is not None
        assert result.y_pred is not None
        assert len(result.y_test) == len(result.y_pred)


# ---------------------------------------------------------------------------
# Clustering training
# ---------------------------------------------------------------------------

class TestClusteringTraining:
    def test_kmeans(self, lab: MLLab, clustering_df: pd.DataFrame) -> None:
        result = lab.train(
            clustering_df, "a", ["a", "b"],
            "clustering", "KMeans", {"n_clusters": "3"}, 0.2,
        )
        assert "silhouette" in result.metrics
        assert "inertia" in result.metrics
        assert result.metrics["n_clusters"] == 3

    def test_dbscan(self, lab: MLLab, clustering_df: pd.DataFrame) -> None:
        result = lab.train(
            clustering_df, "a", ["a", "b"],
            "clustering", "DBSCAN", {"eps": "2.0", "min_samples": "3"}, 0.2,
        )
        assert "n_clusters" in result.metrics
        assert result.y_pred is not None


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

class TestParamParsing:
    def test_int(self) -> None:
        assert MLLab._parse_param("42", "int") == 42

    def test_float(self) -> None:
        assert MLLab._parse_param("3.14", "float") == pytest.approx(3.14)

    def test_int_or_none_value(self) -> None:
        assert MLLab._parse_param("10", "int_or_none") == 10

    def test_int_or_none_none(self) -> None:
        assert MLLab._parse_param("None", "int_or_none") is None

    def test_choice(self) -> None:
        assert MLLab._parse_param("rbf", "choice") == "rbf"


# ---------------------------------------------------------------------------
# Export / load model
# ---------------------------------------------------------------------------

class TestExportModel:
    def test_export_and_load(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2"],
            "classification", "Logistic Regression", {}, 0.3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            MLLab.export_model(result.model, path)
            assert path.exists()
            loaded = MLLab.load_model(path)
            # Loaded model should predict the same
            preds = loaded.predict(classification_df[["f1", "f2"]])
            assert len(preds) == len(classification_df)


# ---------------------------------------------------------------------------
# TrainResult serialisation
# ---------------------------------------------------------------------------

class TestTrainResultSerialisation:
    def test_to_dict(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        result = lab.train(
            classification_df, "target", ["f1", "f2"],
            "classification", "Logistic Regression", {}, 0.3,
        )
        d = result.to_dict()
        assert d["task"] == "classification"
        assert d["algorithm"] == "Logistic Regression"
        assert "metrics" in d
        assert "feature_names" in d


# ---------------------------------------------------------------------------
# Results accumulation
# ---------------------------------------------------------------------------

class TestResultsAccumulation:
    def test_results_list_grows(self, lab: MLLab, classification_df: pd.DataFrame) -> None:
        assert len(lab.results) == 0
        lab.train(classification_df, "target", ["f1"], "classification", "KNN", {}, 0.3)
        assert len(lab.results) == 1
        lab.train(classification_df, "target", ["f1"], "classification", "KNN", {}, 0.3)
        assert len(lab.results) == 2
