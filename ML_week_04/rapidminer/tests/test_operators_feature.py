"""
test_operators_feature.py – Tests for feature engineering operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.operator_base import get_operator_class


class TestPCA:
    def test_dimension_reduction(self, iris_df):
        op = get_operator_class("PCA")()
        op.set_param("n_components", 2)
        result = op.execute({"in": iris_df})
        out = result["out"]
        pc_cols = [c for c in out.columns if c.startswith("PC")]
        assert len(pc_cols) == 2
        assert "model" in result

    def test_default_components(self, iris_df):
        op = get_operator_class("PCA")()
        result = op.execute({"in": iris_df})
        pc_cols = [c for c in result["out"].columns if c.startswith("PC")]
        assert len(pc_cols) >= 1


class TestVarianceThreshold:
    def test_removes_low_variance(self):
        df = pd.DataFrame({
            "constant": [1] * 50,
            "varies": list(range(50)),
        })
        op = get_operator_class("Variance Threshold")()
        op.set_param("threshold", 0.01)
        out = op.execute({"in": df})["out"]
        assert "constant" not in out.columns
        assert "varies" in out.columns


class TestCorrelationMatrix:
    def test_output_shape(self, iris_df):
        op = get_operator_class("Correlation Matrix")()
        op.set_param("method", "pearson")
        out = op.execute({"in": iris_df})["out"]
        num_cols = iris_df.select_dtypes(include="number").columns
        # Output has a "feature" column + one col per numeric feature
        assert out.shape[0] == len(num_cols)
        assert "feature" in out.columns or out.shape[1] == len(num_cols)


class TestWeightByCorrelation:
    def test_returns_weights(self, regression_df):
        # Needs a numeric label to compute correlations
        op = get_operator_class("Weight by Correlation")()
        out = op.execute({"in": regression_df})["out"]
        assert "feature" in out.columns
        assert "weight" in out.columns
        assert len(out) > 0


class TestOneHotEncoding:
    def test_basic(self, iris_df):
        op = get_operator_class("One Hot Encoding")()
        op.set_param("columns", "species")
        out = op.execute({"in": iris_df})["out"]
        assert "species" not in out.columns or out.shape[1] > iris_df.shape[1]
        # Should have dummy columns
        species_cols = [c for c in out.columns if "species" in c.lower() or "setosa" in c.lower() or "versicolor" in c.lower() or "virginica" in c.lower()]
        assert len(species_cols) >= 2


class TestLabelEncoding:
    def test_basic(self, iris_df):
        op = get_operator_class("Label Encoding")()
        op.set_param("columns", "species")
        out = op.execute({"in": iris_df})["out"]
        assert out["species"].dtype in (np.int64, np.int32, int)


class TestTargetEncoding:
    def test_basic(self, iris_df):
        # Need a numeric label for target encoding
        df = iris_df.copy()
        df["target_num"] = np.random.RandomState(0).randint(0, 2, len(df))
        df.attrs["_roles"] = {"target_num": "label"}
        op = get_operator_class("Target Encoding")()
        op.set_param("columns", "species")
        out = op.execute({"in": df})["out"]
        assert out["species"].dtype == np.float64 or np.issubdtype(out["species"].dtype, np.floating)
