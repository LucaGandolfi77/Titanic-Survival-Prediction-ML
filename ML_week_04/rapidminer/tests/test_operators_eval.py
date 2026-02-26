"""
test_operators_eval.py – Tests for evaluation operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.operator_base import get_operator_class


# ═══════════════════════════════════════════════════════════════════════════
# Apply Model
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyModel:
    def test_classification(self, iris_df):
        # Train a model first
        rf = get_operator_class("Random Forest")()
        rf.set_param("n_estimators", 10)
        train_result = rf.execute({"training": iris_df})
        model = train_result["model"]

        # Apply to same data
        apply_op = get_operator_class("Apply Model")()
        result = apply_op.execute({"model": model, "unlabelled": iris_df})
        df_out = result["out"]
        assert "prediction" in df_out.columns
        assert len(df_out) == len(iris_df)

    def test_confidence_columns(self, iris_df):
        lr = get_operator_class("Logistic Regression")()
        lr.set_param("max_iter", 200)
        model = lr.execute({"training": iris_df})["model"]

        apply_op = get_operator_class("Apply Model")()
        df_out = apply_op.execute({"model": model, "unlabelled": iris_df})["out"]
        conf_cols = [c for c in df_out.columns if c.startswith("confidence_")]
        assert len(conf_cols) == 3  # 3 species

    def test_regression(self, regression_df):
        lin = get_operator_class("Linear Regression")()
        model = lin.execute({"training": regression_df})["model"]

        apply_op = get_operator_class("Apply Model")()
        df_out = apply_op.execute({"model": model, "unlabelled": regression_df})["out"]
        assert "prediction" in df_out.columns


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Classification)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceClassification:
    def test_metrics(self, iris_df):
        # Build a df with prediction column
        rf = get_operator_class("Random Forest")()
        rf.set_param("n_estimators", 10)
        model = rf.execute({"training": iris_df})["model"]

        apply_op = get_operator_class("Apply Model")()
        labelled = apply_op.execute({"model": model, "unlabelled": iris_df})["out"]

        perf_op = get_operator_class("Performance (Classification)")()
        result = perf_op.execute({"in": labelled})
        perf = result["performance"]
        assert "accuracy" in perf
        assert "precision" in perf
        assert "recall" in perf
        assert "f1" in perf
        assert "confusion_matrix" in perf
        assert perf["accuracy"] > 0

    def test_missing_prediction_raises(self, iris_df):
        perf_op = get_operator_class("Performance (Classification)")()
        with pytest.raises(ValueError, match="prediction"):
            perf_op.execute({"in": iris_df})


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Regression)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceRegression:
    def test_metrics(self, regression_df):
        lin = get_operator_class("Linear Regression")()
        model = lin.execute({"training": regression_df})["model"]

        apply_op = get_operator_class("Apply Model")()
        labelled = apply_op.execute({"model": model, "unlabelled": regression_df})["out"]

        perf_op = get_operator_class("Performance (Regression)")()
        result = perf_op.execute({"in": labelled})
        perf = result["performance"]
        assert "mae" in perf
        assert "mse" in perf
        assert "rmse" in perf
        assert "r2" in perf
        assert perf["r2"] > 0  # should be a good fit


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Clustering)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceClustering:
    def test_metrics(self, iris_df):
        km = get_operator_class("KMeans")()
        km.set_param("n_clusters", 3)
        train_result = km.execute({"training": iris_df})
        clustered = train_result["out"]
        model = train_result["model"]

        perf_op = get_operator_class("Performance (Clustering)")()
        result = perf_op.execute({"in": clustered, "model": model})
        perf = result["performance"]
        assert "silhouette" in perf
        assert perf["silhouette"] is not None
        assert "cluster_distribution" in perf

    def test_missing_cluster_raises(self, iris_df):
        perf_op = get_operator_class("Performance (Clustering)")()
        with pytest.raises(ValueError, match="cluster"):
            perf_op.execute({"in": iris_df})


# ═══════════════════════════════════════════════════════════════════════════
# Cross Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    def test_basic(self, iris_df):
        op = get_operator_class("Cross Validation")()
        op.set_param("k", 3)
        op.set_param("estimator", "Decision Tree")
        op.set_param("scoring", "accuracy")
        result = op.execute({"in": iris_df})
        perf = result["performance"]
        assert perf["type"] == "cross_validation"
        assert perf["k"] == 3
        assert len(perf["scores"]) == 3
        assert 0 <= perf["mean"] <= 1


# ═══════════════════════════════════════════════════════════════════════════
# Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureImportance:
    def test_tree_model(self, iris_df):
        rf = get_operator_class("Random Forest")()
        rf.set_param("n_estimators", 10)
        model = rf.execute({"training": iris_df})["model"]

        fi = get_operator_class("Feature Importance")()
        out = fi.execute({"model": model})["out"]
        assert "feature" in out.columns
        assert "importance" in out.columns
        assert len(out) > 0

    def test_linear_model(self, regression_df):
        lin = get_operator_class("Linear Regression")()
        model = lin.execute({"training": regression_df})["model"]

        fi = get_operator_class("Feature Importance")()
        out = fi.execute({"model": model})["out"]
        assert "feature" in out.columns
        assert len(out) > 0
