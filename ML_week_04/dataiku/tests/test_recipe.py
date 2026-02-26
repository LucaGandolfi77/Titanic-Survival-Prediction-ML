"""Tests for core/recipe.py"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from core.recipe import (
    RecipeEngine,
    STEP_HANDLERS,
    new_prepare_step,
    new_recipe_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, None, 40, 45],
        "salary": [30000, 40000, 50000, 60000, 70000],
        "city": ["NY", "LA", "NY", "LA", "NY"],
        "date": pd.to_datetime(["2021-01-15", "2021-06-20", "2022-03-10", "2022-11-05", "2023-07-30"]),
    })


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

class TestStepFilterRows:
    def test_filter_numeric_eq(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("filter_rows", column="age", operator="==", value="30")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert len(result) == 1
        assert result.iloc[0]["age"] == 30

    def test_filter_numeric_gt(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("filter_rows", column="age", operator=">", value="35")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert all(result["age"] > 35)

    def test_filter_string_eq(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("filter_rows", column="city", operator="==", value="NY")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert len(result) == 3

    def test_filter_contains(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("filter_rows", column="city", operator="contains", value="N")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert len(result) == 3


class TestStepDropColumns:
    def test_drop_single(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("drop_columns", columns=["city"])
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert "city" not in result.columns
        assert "age" in result.columns

    def test_drop_multiple(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("drop_columns", columns=["city", "salary"])
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert "city" not in result.columns
        assert "salary" not in result.columns

    def test_drop_nonexistent_silent(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("drop_columns", columns=["nonexistent"])
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert len(result.columns) == len(sample_df.columns)


class TestStepRenameColumn:
    def test_rename(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("rename_column", old_name="age", new_name="years")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert "years" in result.columns
        assert "age" not in result.columns


class TestStepFillNA:
    def test_fill_constant(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("fill_na", column="age", method="constant", value=99)
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert result["age"].isna().sum() == 0

    def test_fill_mean(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("fill_na", column="age", method="mean")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert result["age"].isna().sum() == 0

    def test_fill_median(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("fill_na", column="age", method="median")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert result["age"].isna().sum() == 0

    def test_fill_mode(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("fill_na", column="age", method="mode")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert result["age"].isna().sum() == 0


class TestStepEncode:
    def test_label_encode(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("label_encode", column="city")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert pd.api.types.is_numeric_dtype(result["city"])

    def test_onehot_encode(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("onehot_encode", column="city")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert "city" not in result.columns
        assert any("city_" in c for c in result.columns)


class TestStepNormalize:
    def test_normalize(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("normalize", column="salary")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert pytest.approx(result["salary"].min()) == 0.0
        assert pytest.approx(result["salary"].max()) == 1.0


class TestStepStandardize:
    def test_standardize(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("standardize", column="salary")
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert pytest.approx(result["salary"].mean(), abs=1e-10) == 0.0
        assert pytest.approx(result["salary"].std(), abs=0.1) == 1.0


class TestStepExtractDatetime:
    def test_extract_features(self, sample_df: pd.DataFrame) -> None:
        step = new_prepare_step("extract_datetime", column="date", features=["year", "month", "day", "weekday"])
        result = RecipeEngine.execute_prepare(sample_df, [step])
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_day" in result.columns
        assert "date_weekday" in result.columns
        assert result["date_year"].iloc[0] == 2021


class TestStepCustomFormula:
    def test_formula(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        step = new_prepare_step("custom_formula", new_column="c", formula="a + b")
        result = RecipeEngine.execute_prepare(df, [step])
        assert "c" in result.columns
        assert list(result["c"]) == [11, 22, 33]


# ---------------------------------------------------------------------------
# Multiple steps
# ---------------------------------------------------------------------------

class TestMultipleSteps:
    def test_chained_steps(self, sample_df: pd.DataFrame) -> None:
        steps = [
            new_prepare_step("fill_na", column="age", method="mean"),
            new_prepare_step("normalize", column="salary"),
            new_prepare_step("drop_columns", columns=["date"]),
        ]
        result = RecipeEngine.execute_prepare(sample_df, steps)
        assert result["age"].isna().sum() == 0
        assert pytest.approx(result["salary"].max()) == 1.0
        assert "date" not in result.columns


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

class TestPreview:
    def test_preview_returns_limited_rows(self, sample_df: pd.DataFrame) -> None:
        steps = [new_prepare_step("fill_na", column="age", method="mean")]
        preview = RecipeEngine.preview_prepare(sample_df, steps, n_rows=3)
        assert len(preview) <= 3


# ---------------------------------------------------------------------------
# Join recipe
# ---------------------------------------------------------------------------

class TestJoinRecipe:
    def test_inner_join(self) -> None:
        left = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"id": [2, 3, 4], "score": [90, 80, 70]})
        result = RecipeEngine.execute_join(left, right, how="inner", left_on="id", right_on="id")
        assert len(result) == 2
        assert set(result["id"]) == {2, 3}

    def test_left_join(self) -> None:
        left = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"id": [2, 3, 4], "score": [90, 80, 70]})
        result = RecipeEngine.execute_join(left, right, how="left", left_on="id", right_on="id")
        assert len(result) == 3

    def test_outer_join(self) -> None:
        left = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        right = pd.DataFrame({"id": [2, 3], "score": [90, 80]})
        result = RecipeEngine.execute_join(left, right, how="outer", left_on="id", right_on="id")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# GroupBy recipe
# ---------------------------------------------------------------------------

class TestGroupByRecipe:
    def test_groupby_sum(self) -> None:
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [10, 20, 30, 40]})
        result = RecipeEngine.execute_groupby(df, group_cols=["cat"], agg_dict={"val": ["sum", "mean"]})
        assert "val_sum" in result.columns
        assert "val_mean" in result.columns
        assert len(result) == 2

    def test_groupby_multiple_cols(self) -> None:
        df = pd.DataFrame({
            "cat": ["A", "A", "B", "B"],
            "sub": ["x", "y", "x", "y"],
            "val": [1, 2, 3, 4],
        })
        result = RecipeEngine.execute_groupby(df, group_cols=["cat", "sub"], agg_dict={"val": ["sum"]})
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Python recipe
# ---------------------------------------------------------------------------

class TestPythonRecipe:
    def test_custom_code(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = "output_df = df.assign(b=df['a'] * 10)"
        result = RecipeEngine.execute_python(df, code)
        assert "b" in result.columns
        assert list(result["b"]) == [10, 20, 30]

    def test_no_output_df(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = "df['b'] = 42"
        result = RecipeEngine.execute_python(df, code)
        assert "b" in result.columns

    def test_original_not_mutated(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = "df['b'] = 99"
        RecipeEngine.execute_python(df, code)
        assert "b" not in df.columns  # original unchanged


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

class TestExecuteRecipe:
    def test_dispatch_prepare(self) -> None:
        df = pd.DataFrame({"x": [1, 2, None]})
        config = {
            "input": "ds1",
            "steps": [new_prepare_step("fill_na", column="x", method="mean")],
        }
        result = RecipeEngine.execute_recipe("prepare", config, {"ds1": df})
        assert result["x"].isna().sum() == 0

    def test_dispatch_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown recipe type"):
            RecipeEngine.execute_recipe("magic", {}, {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestNewRecipeConfig:
    def test_creates_id(self) -> None:
        cfg = new_recipe_config("prepare", input_name="ds1", output_name="out")
        assert "id" in cfg
        assert cfg["type"] == "prepare"
        assert cfg["input"] == "ds1"

    def test_step_handlers_registered(self) -> None:
        expected = {
            "filter_rows", "drop_columns", "rename_column", "fill_na",
            "label_encode", "onehot_encode", "normalize", "standardize",
            "extract_datetime", "custom_formula",
        }
        assert expected == set(STEP_HANDLERS.keys())
