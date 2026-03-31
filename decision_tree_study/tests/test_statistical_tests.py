"""Tests for evaluation.statistical_tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.statistical_tests import (
    cohens_d,
    correlation_analysis,
    friedman_test,
    nemenyi_critical_difference,
    pairwise_wilcoxon,
    wilcoxon_test,
)


class TestCohensD:
    def test_identical_samples(self):
        a = np.ones(20)
        assert cohens_d(a, a) == 0.0

    def test_known_effect(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100) + 2.0
        d = cohens_d(b, a)
        # b is shifted up by 2, so d should be large positive
        assert d > 1.0

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50) + 1
        assert abs(cohens_d(a, b) + cohens_d(b, a)) < 1e-10


class TestWilcoxonTest:
    def test_identical_scores(self):
        a = np.array([0.8, 0.85, 0.9, 0.88, 0.82])
        result = wilcoxon_test(a, a)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_different_scores(self):
        a = np.array([0.9, 0.92, 0.91, 0.93, 0.89, 0.90, 0.91, 0.92, 0.93, 0.90])
        b = np.array([0.7, 0.72, 0.71, 0.73, 0.69, 0.70, 0.71, 0.72, 0.73, 0.70])
        result = wilcoxon_test(a, b)
        assert result["p_value"] < 0.05
        assert result["significant"] == True

    def test_returns_cohens_d(self):
        a = np.array([0.8, 0.85, 0.9])
        b = np.array([0.7, 0.75, 0.8])
        result = wilcoxon_test(a, b)
        assert "cohens_d" in result


class TestFriedmanTest:
    def test_similar_groups(self):
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((20, 3)) + 5  # all similar
        result = friedman_test(matrix, ["a", "b", "c"])
        assert "statistic" in result
        assert "avg_ranks" in result
        assert len(result["avg_ranks"]) == 3

    def test_different_groups(self):
        # Group 3 is consistently best
        matrix = np.column_stack([
            np.full(20, 0.7),
            np.full(20, 0.8),
            np.full(20, 0.9),
        ])
        # Add a tiny bit of noise so Friedman can work
        rng = np.random.default_rng(42)
        matrix += rng.standard_normal(matrix.shape) * 0.01
        result = friedman_test(matrix, ["low", "mid", "high"])
        assert result["significant"] == True


class TestNemenyiCD:
    def test_positive(self):
        cd = nemenyi_critical_difference(10, 5)
        assert cd > 0

    def test_increases_with_k(self):
        cd3 = nemenyi_critical_difference(10, 3)
        cd5 = nemenyi_critical_difference(10, 5)
        assert cd5 > cd3


class TestCorrelationAnalysis:
    def test_perfect_positive(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = correlation_analysis(df, "x", "y")
        assert abs(result["pearson_r"] - 1.0) < 1e-10

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(200), "y": rng.standard_normal(200)})
        result = correlation_analysis(df, "x", "y")
        assert abs(result["pearson_r"]) < 0.2


class TestPairwiseWilcoxon:
    def test_returns_dataframe(self):
        data = {
            "strategy": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "seed": list(range(10)) * 3,
            "test_accuracy": np.concatenate([
                np.full(10, 0.9), np.full(10, 0.8), np.full(10, 0.7),
            ]) + np.random.default_rng(42).standard_normal(30) * 0.01,
        }
        df = pd.DataFrame(data)
        result = pairwise_wilcoxon(df)
        assert len(result) == 3  # C(3,2) = 3 pairs
        assert "strategy_a" in result.columns
        assert "strategy_b" in result.columns
        assert "p_value" in result.columns
