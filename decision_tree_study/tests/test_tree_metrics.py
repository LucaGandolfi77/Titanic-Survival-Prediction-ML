"""Tests for trees.tree_metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trees.tree_metrics import compute_metrics, timed_fit


@pytest.fixture
def fitted_tree_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_tr, y_tr)
    return tree, X_tr, y_tr, X_val, y_val, X_te, y_te


class TestComputeMetrics:
    def test_returns_expected_keys(self, fitted_tree_data):
        tree, X_tr, y_tr, X_val, y_val, X_te, y_te = fitted_tree_data
        m = compute_metrics(tree, X_tr, y_tr, X_val, y_val, X_te, y_te)
        expected_keys = {
            "train_accuracy", "val_accuracy", "test_accuracy",
            "train_f1", "val_f1", "test_f1",
            "tree_depth", "n_leaves", "n_nodes",
            "interpretability_score", "overfitting_gap",
        }
        assert expected_keys.issubset(m.keys())

    def test_accuracy_range(self, fitted_tree_data):
        tree, X_tr, y_tr, X_val, y_val, X_te, y_te = fitted_tree_data
        m = compute_metrics(tree, X_tr, y_tr, X_val, y_val, X_te, y_te)
        for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
            assert 0.0 <= m[key] <= 1.0

    def test_overfitting_gap_correct(self, fitted_tree_data):
        tree, X_tr, y_tr, X_val, y_val, X_te, y_te = fitted_tree_data
        m = compute_metrics(tree, X_tr, y_tr, X_val, y_val, X_te, y_te)
        expected_gap = m["train_accuracy"] - m["test_accuracy"]
        assert abs(m["overfitting_gap"] - expected_gap) < 1e-10

    def test_interpretability_score(self, fitted_tree_data):
        tree, X_tr, y_tr, X_val, y_val, X_te, y_te = fitted_tree_data
        m = compute_metrics(tree, X_tr, y_tr, X_val, y_val, X_te, y_te)
        expected = 1.0 / (1.0 + m["n_leaves"])
        assert abs(m["interpretability_score"] - expected) < 1e-10

    def test_train_accuracy_is_one_for_deep_tree(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        m = compute_metrics(tree, X, y, X, y, X, y)
        assert m["train_accuracy"] == 1.0


class TestTimedFit:
    def test_returns_positive_time(self):
        X, y = make_classification(n_samples=100, random_state=42)
        tree = DecisionTreeClassifier(random_state=42)
        ms = timed_fit(tree, X, y)
        assert ms > 0.0

    def test_tree_is_fitted(self):
        X, y = make_classification(n_samples=100, random_state=42)
        tree = DecisionTreeClassifier(random_state=42)
        timed_fit(tree, X, y)
        assert tree.tree_ is not None
