"""Tests for src.data.loader"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import (
    get_dataset_summary,
    list_sample_datasets,
    load_sample_dataset,
)


class TestListSampleDatasets:
    def test_returns_list(self):
        result = list_sample_datasets()
        assert isinstance(result, list)

    def test_contains_iris(self):
        result = list_sample_datasets()
        assert "iris.csv" in result

    def test_sorted(self):
        result = list_sample_datasets()
        assert result == sorted(result)


class TestLoadSampleDataset:
    def test_load_iris(self):
        df = load_sample_dataset("iris.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] >= 2

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sample_dataset("nonexistent.csv")

    def test_columns_present(self):
        df = load_sample_dataset("iris.csv")
        assert "target" in df.columns


class TestGetDatasetSummary:
    def test_keys(self):
        df = load_sample_dataset("iris.csv")
        summary = get_dataset_summary(df)
        expected_keys = {"rows", "columns", "numeric_cols", "categorical_cols", "missing_cells", "missing_pct", "duplicated_rows"}
        assert expected_keys.issubset(summary.keys())

    def test_iris_no_missing(self):
        df = load_sample_dataset("iris.csv")
        summary = get_dataset_summary(df)
        assert summary["missing_cells"] == 0
