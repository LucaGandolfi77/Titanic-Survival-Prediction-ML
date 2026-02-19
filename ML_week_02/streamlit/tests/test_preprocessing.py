"""Tests for src.data.preprocessing"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    encode_categoricals,
    impute_missing,
    preprocess,
    scale_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, np.nan, 35, 45, np.nan],
        "salary": [50000, 60000, np.nan, 80000, 90000],
        "city": ["NY", "LA", np.nan, "NY", "SF"],
        "target": [0, 1, 0, 1, 0],
    })


class TestImputeMissing:
    def test_no_nans_after(self, sample_df):
        result = impute_missing(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_shape_preserved(self, sample_df):
        result = impute_missing(sample_df)
        assert result.shape == sample_df.shape


class TestEncodeCategoricals:
    def test_one_hot_no_object_cols(self, sample_df):
        df = impute_missing(sample_df)
        result = encode_categoricals(df, strategy="one_hot", target_col="target")
        assert result.select_dtypes("object").shape[1] == 0

    def test_label_encoding(self, sample_df):
        df = impute_missing(sample_df)
        result = encode_categoricals(df, strategy="label", target_col="target")
        assert result["city"].dtype in [np.int64, np.int32, int]


class TestScaleFeatures:
    def test_standard_scaling(self, sample_df):
        df = impute_missing(sample_df)
        df = encode_categoricals(df, target_col="target")
        result = scale_features(df, strategy="standard", target_col="target")
        # Numeric columns should be roughly mean-0
        num_cols = [c for c in result.select_dtypes("number").columns if c != "target"]
        for c in num_cols:
            assert abs(result[c].mean()) < 0.1

    def test_none_scaling_no_change(self, sample_df):
        df = impute_missing(sample_df)
        result = scale_features(df, strategy="none")
        pd.testing.assert_frame_equal(df, result)


class TestPreprocess:
    def test_full_pipeline(self, sample_df):
        result = preprocess(sample_df, target_col="target")
        assert result.isnull().sum().sum() == 0
        assert result.select_dtypes("object").shape[1] == 0
