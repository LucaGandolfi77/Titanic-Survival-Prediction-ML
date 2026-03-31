"""Tests for evaluation.crossval."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.crossval import cv_summary, multi_seed_cv


@pytest.fixture
def iris_data():
    data = load_iris()
    return data.data, data.target


class TestMultiSeedCV:
    def test_returns_dataframe(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=3, seeds=[42])
        assert len(df) == 3  # 3 folds × 1 seed

    def test_columns(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=3, seeds=[42])
        expected = {"seed", "fold", "train_accuracy", "val_accuracy",
                    "train_f1", "val_f1", "n_leaves", "tree_depth"}
        assert expected.issubset(set(df.columns))

    def test_multi_seed(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=3, seeds=[42, 7])
        assert len(df) == 6  # 3 folds × 2 seeds

    def test_accuracy_range(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=5, seeds=[42])
        assert df["val_accuracy"].between(0, 1).all()

    def test_with_tree_kwargs(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, tree_kwargs={"max_depth": 2}, n_folds=3, seeds=[42])
        assert (df["tree_depth"] <= 2).all()


class TestCVSummary:
    def test_keys(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=3, seeds=[42])
        s = cv_summary(df)
        expected = {"mean_val_accuracy", "std_val_accuracy", "mean_val_f1",
                    "mean_train_accuracy", "mean_n_leaves", "mean_tree_depth",
                    "n_evaluations"}
        assert expected == set(s.keys())

    def test_n_evaluations(self, iris_data):
        X, y = iris_data
        df = multi_seed_cv(X, y, n_folds=5, seeds=[42, 7])
        s = cv_summary(df)
        assert s["n_evaluations"] == 10
