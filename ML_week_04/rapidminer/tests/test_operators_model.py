"""
test_operators_model.py – Tests for ML model operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.operator_base import get_operator_class


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _train(op_type: str, df: pd.DataFrame, **params) -> dict:
    """Convenience: instantiate, set params, execute, return outputs."""
    op = get_operator_class(op_type)()
    for k, v in params.items():
        op.set_param(k, v)
    return op.execute({"training": df})


# ═══════════════════════════════════════════════════════════════════════════
# Classification models
# ═══════════════════════════════════════════════════════════════════════════

class TestLogisticRegression:
    def test_train(self, iris_df):
        result = _train("Logistic Regression", iris_df, max_iter=200)
        assert "model" in result
        assert "training_performance" in result
        assert hasattr(result["model"], "predict")

    def test_accuracy_positive(self, iris_df):
        result = _train("Logistic Regression", iris_df)
        assert result["training_performance"]["accuracy_train"] > 0


class TestDecisionTree:
    def test_train(self, iris_df):
        result = _train("Decision Tree", iris_df, max_depth=3)
        assert hasattr(result["model"], "predict")

    def test_feature_importances(self, iris_df):
        result = _train("Decision Tree", iris_df)
        assert hasattr(result["model"], "feature_importances_")


class TestRandomForest:
    def test_train(self, iris_df):
        result = _train("Random Forest", iris_df, n_estimators=10)
        assert hasattr(result["model"], "predict")
        assert result["training_performance"]["accuracy_train"] > 0

    def test_n_estimators(self, iris_df):
        result = _train("Random Forest", iris_df, n_estimators=5)
        assert len(result["model"].estimators_) == 5


class TestGradientBoosting:
    def test_train(self, iris_df):
        result = _train("Gradient Boosting", iris_df, n_estimators=10)
        assert hasattr(result["model"], "predict")


class TestSVM:
    def test_train(self, iris_df):
        result = _train("SVM", iris_df)
        assert hasattr(result["model"], "predict")


class TestKNN:
    def test_train(self, iris_df):
        result = _train("KNN", iris_df, n_neighbors=3)
        assert hasattr(result["model"], "predict")


class TestNaiveBayes:
    def test_train(self, iris_df):
        result = _train("Naive Bayes", iris_df)
        assert hasattr(result["model"], "predict")


# ═══════════════════════════════════════════════════════════════════════════
# Regression models
# ═══════════════════════════════════════════════════════════════════════════

class TestLinearRegression:
    def test_train(self, regression_df):
        result = _train("Linear Regression", regression_df)
        assert hasattr(result["model"], "predict")
        assert "training_performance" in result


class TestRidge:
    def test_train(self, regression_df):
        result = _train("Ridge", regression_df, alpha=1.0)
        assert hasattr(result["model"], "coef_")


class TestLasso:
    def test_train(self, regression_df):
        result = _train("Lasso", regression_df, alpha=0.1)
        assert hasattr(result["model"], "coef_")


# ═══════════════════════════════════════════════════════════════════════════
# Clustering models
# ═══════════════════════════════════════════════════════════════════════════

class TestKMeans:
    def test_train(self, iris_df):
        result = _train("KMeans", iris_df, n_clusters=3)
        assert "model" in result
        assert "out" in result  # returns labelled data with "cluster" col
        assert "cluster" in result["out"].columns

    def test_n_clusters(self, iris_df):
        result = _train("KMeans", iris_df, n_clusters=2)
        assert result["out"]["cluster"].nunique() == 2


class TestDBSCAN:
    def test_train(self, iris_df):
        result = _train("DBSCAN", iris_df)
        assert "cluster" in result["out"].columns


class TestAgglomerative:
    def test_train(self, iris_df):
        result = _train("Agglomerative", iris_df, n_clusters=3)
        assert "cluster" in result["out"].columns
        assert result["out"]["cluster"].nunique() == 3
