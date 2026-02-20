"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from automl.features.numeric import NumericFeatureEngineer
from automl.features.categorical import CategoricalFeatureEngineer
from automl.features.datetime_features import DateTimeFeatureEngineer
from automl.features.text_features import TextFeatureEngineer
from automl.features.selector import FeatureSelector


class TestNumericEngineer:

    def test_fit_transform_shape(self, config):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        eng = NumericFeatureEngineer(config.features.numeric)
        arr = eng.fit_transform(df)
        assert arr.shape[0] == 5
        assert arr.shape[1] >= 2

    def test_transform_matches(self, config):
        df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
        eng = NumericFeatureEngineer(config.features.numeric)
        eng.fit_transform(df)
        arr_new = eng.transform(df)
        assert arr_new.shape[0] == 20


class TestCategoricalEngineer:

    def test_binary_encoding(self, config):
        df = pd.DataFrame({"sex": ["male", "female"] * 10})
        y = pd.Series([0, 1] * 10)
        eng = CategoricalFeatureEngineer(config.features.categorical)
        arr = eng.fit_transform(df, y)
        assert arr.shape == (20, 1)  # label encoded → 1 column

    def test_ohe(self, config):
        df = pd.DataFrame({"color": ["red", "green", "blue", "yellow"] * 5})
        y = pd.Series([0, 1, 0, 1] * 5)
        eng = CategoricalFeatureEngineer(config.features.categorical)
        arr = eng.fit_transform(df, y)
        assert arr.shape[0] == 20
        assert arr.shape[1] >= 3  # at least 3 OHE cols


class TestDateTimeEngineer:

    def test_cyclical(self, config):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=30, freq="D").astype(str)
        })
        eng = DateTimeFeatureEngineer(config.features.datetime)
        arr = eng.fit_transform(df)
        assert arr.shape[0] == 30
        assert arr.shape[1] >= 2  # at least sin + cos for month


class TestTextEngineer:

    def test_tfidf(self, config):
        df = pd.DataFrame({"text": ["hello world", "foo bar baz", "machine learning", "hello foo"]})
        eng = TextFeatureEngineer(config.features.text)
        arr = eng.fit_transform(df)
        assert arr.shape[0] == 4
        assert arr.shape[1] >= 1


class TestFeatureSelector:

    def test_reduces_features(self, config):
        rng = np.random.RandomState(42)
        n, p = 100, 20
        X = rng.randn(n, p)
        # Make some features near-constant
        X[:, 18] = 0.0
        X[:, 19] = 0.0
        y = pd.Series(rng.choice([0, 1], n))
        sel = FeatureSelector(config.features.selection)
        X_out = sel.fit_transform(X, y)
        assert X_out.shape[1] <= p
