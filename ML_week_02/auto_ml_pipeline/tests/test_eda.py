"""Tests for EDA modules."""

import numpy as np
import pandas as pd
import pytest

from automl.eda.type_detector import TypeDetector
from automl.eda.missing_handler import MissingValueHandler
from automl.eda.outlier_detector import OutlierDetector
from automl.eda.profiler import DataProfiler


class TestTypeDetector:

    def test_detect_numeric(self, binary_clf_df):
        det = TypeDetector()
        types = det.detect(binary_clf_df.drop(columns=["Survived"]))
        assert "Age" in types["numeric"]
        assert "Fare" in types["numeric"]

    def test_detect_categorical(self, binary_clf_df):
        det = TypeDetector()
        types = det.detect(binary_clf_df.drop(columns=["Survived"]))
        assert "Sex" in types["categorical"]

    def test_all_columns_assigned(self, binary_clf_df):
        det = TypeDetector()
        types = det.detect(binary_clf_df.drop(columns=["Survived"]))
        all_typed = []
        for v in types.values():
            all_typed.extend(v)
        # Every column appears somewhere
        for col in binary_clf_df.columns:
            if col != "Survived":
                assert col in all_typed, f"{col} not classified"


class TestMissingHandler:

    def test_fills_missing(self, binary_clf_df, config):
        df = binary_clf_df.copy()
        df.loc[0:5, "Age"] = np.nan
        handler = MissingValueHandler(config.eda.missing)
        det = TypeDetector()
        col_types = det.detect(df.drop(columns=["Survived"]))
        result = handler.fit_transform(df.drop(columns=["Survived"]), col_types)
        assert result["Age"].isna().sum() == 0


class TestOutlierDetector:

    def test_outlier_detection(self, binary_clf_df, config):
        det = OutlierDetector(config.eda.outliers)
        num_cols = ["Age", "Fare"]
        result = det.fit_transform(binary_clf_df.drop(columns=["Survived"]), num_cols)
        assert len(result) > 0


class TestProfiler:

    def test_profile_runs(self, binary_clf_df, config):
        profiler = DataProfiler(config.eda)
        report = profiler.profile(binary_clf_df)
        assert "shape" in report
        assert "dtypes" in report
        assert "numeric_stats" in report
        assert "missing_pct" in report
