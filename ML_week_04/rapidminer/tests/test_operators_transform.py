"""
test_operators_transform.py – Tests for the 17 transform operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.operator_base import get_operator_class


# ═══════════════════════════════════════════════════════════════════════════
# Select Attributes
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectAttributes:
    def test_keep(self, iris_df):
        op = get_operator_class("Select Attributes")()
        op.set_param("mode", "keep")
        op.set_param("columns", "sepal_length,sepal_width")
        df = op.execute({"in": iris_df})["out"]
        assert list(df.columns) == ["sepal_length", "sepal_width"]

    def test_remove(self, iris_df):
        op = get_operator_class("Select Attributes")()
        op.set_param("mode", "remove")
        op.set_param("columns", "species")
        df = op.execute({"in": iris_df})["out"]
        assert "species" not in df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Filter Examples
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterExamples:
    def test_simple_filter(self, iris_df):
        op = get_operator_class("Filter Examples")()
        op.set_param("condition", "sepal_length > 6")
        df = op.execute({"in": iris_df})["out"]
        assert all(df["sepal_length"] > 6)

    def test_empty_condition_passthrough(self, iris_df):
        op = get_operator_class("Filter Examples")()
        op.set_param("condition", "")
        df = op.execute({"in": iris_df})["out"]
        assert len(df) == len(iris_df)


# ═══════════════════════════════════════════════════════════════════════════
# Rename
# ═══════════════════════════════════════════════════════════════════════════

class TestRename:
    def test_rename_single(self, iris_df):
        op = get_operator_class("Rename")()
        op.set_param("mapping", "species=label")
        df = op.execute({"in": iris_df})["out"]
        assert "label" in df.columns
        assert "species" not in df.columns

    def test_rename_multiple(self, iris_df):
        op = get_operator_class("Rename")()
        op.set_param("mapping", "sepal_length=sl, sepal_width=sw")
        df = op.execute({"in": iris_df})["out"]
        assert "sl" in df.columns
        assert "sw" in df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Set Role
# ═══════════════════════════════════════════════════════════════════════════

class TestSetRole:
    def test_set_label(self, iris_df):
        op = get_operator_class("Set Role")()
        op.set_param("column", "species")
        op.set_param("role", "label")
        df = op.execute({"in": iris_df})["out"]
        assert df.attrs["_roles"]["species"] == "label"

    def test_set_id(self, iris_df):
        op = get_operator_class("Set Role")()
        op.set_param("column", "sepal_length")
        op.set_param("role", "id")
        df = op.execute({"in": iris_df})["out"]
        assert df.attrs["_roles"]["sepal_length"] == "id"


# ═══════════════════════════════════════════════════════════════════════════
# Replace Missing
# ═══════════════════════════════════════════════════════════════════════════

class TestReplaceMissing:
    def test_mean_imputation(self, mixed_df):
        op = get_operator_class("Replace Missing")()
        op.set_param("strategy", "mean")
        df = op.execute({"in": mixed_df})["out"]
        assert df["age"].isnull().sum() == 0

    def test_constant_imputation(self, mixed_df):
        op = get_operator_class("Replace Missing")()
        op.set_param("strategy", "constant")
        op.set_param("constant", "99")
        op.set_param("columns", "age,salary")
        df = op.execute({"in": mixed_df})["out"]
        assert df["age"].isnull().sum() == 0

    def test_median_imputation(self, mixed_df):
        op = get_operator_class("Replace Missing")()
        op.set_param("strategy", "median")
        df = op.execute({"in": mixed_df})["out"]
        assert df["salary"].isnull().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Normalize
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalize:
    def test_zscore(self, numeric_df):
        op = get_operator_class("Normalize")()
        op.set_param("method", "z-score")
        df = op.execute({"in": numeric_df})["out"]
        assert abs(df["a"].mean()) < 0.01
        assert abs(df["a"].std() - 1.0) < 0.1

    def test_minmax(self, numeric_df):
        op = get_operator_class("Normalize")()
        op.set_param("method", "min-max")
        df = op.execute({"in": numeric_df})["out"]
        assert df["a"].min() >= -0.01
        assert df["a"].max() <= 1.01

    def test_log(self, numeric_df):
        op = get_operator_class("Normalize")()
        op.set_param("method", "log")
        df = op.execute({"in": numeric_df.abs() + 1})["out"]
        assert df["a"].min() > 0


# ═══════════════════════════════════════════════════════════════════════════
# Discretize
# ═══════════════════════════════════════════════════════════════════════════

class TestDiscretize:
    def test_basic(self, iris_df):
        op = get_operator_class("Discretize")()
        op.set_param("column", "sepal_length")
        op.set_param("bins", 3)
        df = op.execute({"in": iris_df})["out"]
        assert df["sepal_length"].dtype.name == "category"


# ═══════════════════════════════════════════════════════════════════════════
# Generate Attributes
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateAttributes:
    def test_formula(self, iris_df):
        op = get_operator_class("Generate Attributes")()
        op.set_param("name", "sepal_ratio")
        op.set_param("formula", "sepal_length / sepal_width")
        df = op.execute({"in": iris_df})["out"]
        assert "sepal_ratio" in df.columns
        assert len(df) == 30


# ═══════════════════════════════════════════════════════════════════════════
# Remove Duplicates
# ═══════════════════════════════════════════════════════════════════════════

class TestRemoveDuplicates:
    def test_remove_dups(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [10, 10, 20, 30]})
        op = get_operator_class("Remove Duplicates")()
        out = op.execute({"in": df})["out"]
        assert len(out) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Sort
# ═══════════════════════════════════════════════════════════════════════════

class TestSort:
    def test_ascending(self, iris_df):
        op = get_operator_class("Sort")()
        op.set_param("column", "sepal_length")
        op.set_param("ascending", True)
        df = op.execute({"in": iris_df})["out"]
        assert list(df["sepal_length"]) == sorted(df["sepal_length"])

    def test_descending(self, iris_df):
        op = get_operator_class("Sort")()
        op.set_param("column", "sepal_length")
        op.set_param("ascending", False)
        df = op.execute({"in": iris_df})["out"]
        assert list(df["sepal_length"]) == sorted(df["sepal_length"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sample
# ═══════════════════════════════════════════════════════════════════════════

class TestSample:
    def test_fraction(self, iris_df):
        op = get_operator_class("Sample")()
        op.set_param("mode", "fraction")
        op.set_param("fraction", 0.5)
        df = op.execute({"in": iris_df})["out"]
        assert len(df) == 15

    def test_absolute(self, iris_df):
        op = get_operator_class("Sample")()
        op.set_param("mode", "absolute")
        op.set_param("n", 10)
        df = op.execute({"in": iris_df})["out"]
        assert len(df) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Split Data
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitData:
    def test_basic_split(self, iris_df):
        op = get_operator_class("Split Data")()
        op.set_param("ratio", 0.7)
        result = op.execute({"in": iris_df})
        assert "train" in result and "test" in result
        assert len(result["train"]) + len(result["test"]) == 30

    def test_ratio_proportions(self, iris_df):
        op = get_operator_class("Split Data")()
        op.set_param("ratio", 0.8)
        result = op.execute({"in": iris_df})
        assert len(result["train"]) == 24  # 80% of 30
        assert len(result["test"]) == 6


# ═══════════════════════════════════════════════════════════════════════════
# Append
# ═══════════════════════════════════════════════════════════════════════════

class TestAppend:
    def test_basic(self, iris_df):
        op = get_operator_class("Append")()
        result = op.execute({"in1": iris_df, "in2": iris_df})["out"]
        assert len(result) == 60


# ═══════════════════════════════════════════════════════════════════════════
# Join
# ═══════════════════════════════════════════════════════════════════════════

class TestJoin:
    def test_inner_join(self):
        left = pd.DataFrame({"key": [1, 2, 3], "val_l": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [2, 3, 4], "val_r": ["x", "y", "z"]})
        op = get_operator_class("Join")()
        op.set_param("key_columns", "key")
        op.set_param("join_type", "inner")
        df = op.execute({"left": left, "right": right})["out"]
        assert len(df) == 2
        assert set(df["key"]) == {2, 3}


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregate:
    def test_groupby_mean(self, iris_df):
        op = get_operator_class("Aggregate")()
        op.set_param("group_by", "species")
        op.set_param("aggregations", "mean")
        df = op.execute({"in": iris_df})["out"]
        assert len(df) == 3  # 3 species


# ═══════════════════════════════════════════════════════════════════════════
# Transpose
# ═══════════════════════════════════════════════════════════════════════════

class TestTranspose:
    def test_transpose(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        op = get_operator_class("Transpose")()
        out = op.execute({"in": df})["out"]
        assert out.shape[0] == 2  # 2 rows (a, b)


# ═══════════════════════════════════════════════════════════════════════════
# Pivot
# ═══════════════════════════════════════════════════════════════════════════

class TestPivot:
    def test_pivot(self):
        df = pd.DataFrame({
            "date": ["Jan", "Jan", "Feb", "Feb"],
            "product": ["A", "B", "A", "B"],
            "sales": [10, 20, 30, 40],
        })
        op = get_operator_class("Pivot")()
        op.set_param("index", "date")
        op.set_param("columns", "product")
        op.set_param("values", "sales")
        op.set_param("aggfunc", "sum")
        out = op.execute({"in": df})["out"]
        assert "A" in out.columns or ("product", "A") in out.columns
