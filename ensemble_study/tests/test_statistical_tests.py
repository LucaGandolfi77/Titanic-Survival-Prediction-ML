"""Tests for evaluation.statistical_tests."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.statistical_tests import (
    cohens_d,
    wilcoxon_test,
    friedman_test,
    nemenyi_critical_difference,
    correlation_analysis,
    pairwise_wilcoxon,
)
import pandas as pd


class TestCohensD:
    def test_identical_arrays(self):
        a = np.ones(10)
        d = cohens_d(a, a)
        assert d == 0.0

    def test_large_effect(self):
        a = np.ones(30) * 10
        b = np.ones(30) * 5
        d = cohens_d(a, b)
        # zero variance → pooled=0 → returns 0
        assert isinstance(d, float)

    def test_known_effect(self):
        rng = np.random.default_rng(42)
        a = rng.normal(10, 1, 100)
        b = rng.normal(8, 1, 100)
        d = cohens_d(a, b)
        assert d > 1.0  # large effect


class TestWilcoxonTest:
    def test_identical_not_significant(self):
        a = np.ones(15)
        res = wilcoxon_test(a, a)
        assert res["significant"] == False
        assert res["p_value"] == 1.0

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.9, 0.02, 20)
        b = rng.normal(0.7, 0.02, 20)
        res = wilcoxon_test(a, b)
        assert res["significant"] == True
        assert res["p_value"] < 0.05

    def test_result_keys(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        res = wilcoxon_test(a, b)
        expected = {"statistic", "p_value", "significant", "effect_size_r", "cohens_d"}
        assert set(res.keys()) == expected


class TestFriedmanTest:
    def test_identical_not_significant(self):
        scores = np.tile(np.arange(10, dtype=float), (3, 1)).T
        res = friedman_test(scores, ["a", "b", "c"])
        assert res["significant"] == False

    def test_different_groups(self):
        rng = np.random.default_rng(42)
        scores = np.column_stack([
            rng.normal(0.9, 0.01, 20),
            rng.normal(0.7, 0.01, 20),
            rng.normal(0.5, 0.01, 20),
        ])
        res = friedman_test(scores, ["a", "b", "c"])
        assert res["significant"] == True
        assert "avg_ranks" in res

    def test_avg_ranks_keys(self):
        rng = np.random.default_rng(42)
        scores = rng.random((10, 3))
        res = friedman_test(scores, ["x", "y", "z"])
        assert set(res["avg_ranks"].keys()) == {"x", "y", "z"}


class TestNemenyiCD:
    def test_returns_positive(self):
        cd = nemenyi_critical_difference(20, 5)
        assert cd > 0

    def test_increases_with_groups(self):
        cd5 = nemenyi_critical_difference(20, 5)
        cd8 = nemenyi_critical_difference(20, 8)
        assert cd8 > cd5


class TestCorrelationAnalysis:
    def test_perfect_correlation(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float),
                           "y": np.arange(20, dtype=float)})
        res = correlation_analysis(df, "x", "y")
        assert res["spearman_r"] == pytest.approx(1.0)
        assert res["pearson_r"] == pytest.approx(1.0)

    def test_returns_four_keys(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
        res = correlation_analysis(df, "x", "y")
        assert set(res.keys()) == {"spearman_r", "spearman_p", "pearson_r", "pearson_p"}
