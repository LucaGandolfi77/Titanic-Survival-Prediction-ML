"""Tests for src.models (registry, trainer, evaluator)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import (
    MODEL_REGISTRY,
    create_estimator,
    get_model,
    get_models_for_task,
)
from src.models.trainer import train_single
from src.models.evaluator import evaluate, comparison_dataframe


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


class TestRegistry:
    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_classification_models_exist(self):
        models = get_models_for_task("classification")
        assert len(models) >= 5

    def test_regression_models_exist(self):
        models = get_models_for_task("regression")
        assert len(models) >= 4

    def test_get_model_known(self):
        info = get_model("logistic_regression")
        assert info.display_name == "Logistic Regression"

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError):
            get_model("nonexistent_model")

    def test_create_estimator(self):
        est = create_estimator("logistic_regression")
        assert hasattr(est, "fit")


class TestTrainer:
    def test_train_single(self, iris_data):
        X, y = iris_data
        result = train_single("logistic_regression", X, y)
        assert result.train_time_sec >= 0
        assert result.estimator is not None

    def test_train_random_forest(self, iris_data):
        X, y = iris_data
        result = train_single("random_forest_clf", X, y, params={"n_estimators": 10})
        preds = result.estimator.predict(X)
        assert len(preds) == len(y)


class TestEvaluator:
    def test_evaluate_classifier(self, iris_data):
        X, y = iris_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tr = train_single("logistic_regression", X_train, y_train)
        er = evaluate(tr, X_test, y_test)
        assert "accuracy" in er.metrics
        assert 0 <= er.metrics["accuracy"] <= 1

    def test_comparison_dataframe(self, iris_data):
        X, y = iris_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tr1 = train_single("logistic_regression", X_train, y_train)
        tr2 = train_single("random_forest_clf", X_train, y_train, params={"n_estimators": 10})
        er1 = evaluate(tr1, X_test, y_test)
        er2 = evaluate(tr2, X_test, y_test)
        comp = comparison_dataframe([er1, er2])
        assert comp.shape[0] == 2
        assert "accuracy" in comp.columns
