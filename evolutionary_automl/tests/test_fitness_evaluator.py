"""Tests for fitness evaluator."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris

from evolutionary_automl.fitness.evaluator import FitnessEvaluator
from evolutionary_automl.fitness.cache import FitnessCache


class TestFitnessEvaluator:
    @pytest.fixture
    def evaluator(self):
        X, y = load_iris(return_X_y=True)
        return FitnessEvaluator(X, y, dataset_name="iris", cv_folds=3, random_state=42)

    def test_evaluate_returns_tuple(self, evaluator):
        chrom = [0.33, 0.0, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        result = evaluator.evaluate(chrom)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_f1_in_valid_range(self, evaluator):
        chrom = [0.33, 0.0, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        f1, _ = evaluator.evaluate(chrom)
        assert 0.0 <= f1 <= 1.0

    def test_caching_works(self, evaluator):
        chrom = [0.33, 0.0, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        r1 = evaluator.evaluate(chrom)
        r2 = evaluator.evaluate(chrom)
        assert r1 == r2
        assert evaluator.cache.hit_rate > 0

    def test_single_objective_wrapper(self, evaluator):
        chrom = [0.5] * 13
        result = evaluator.evaluate_single_objective(chrom)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_multi_objective_wrapper(self, evaluator):
        chrom = [0.5] * 13
        result = evaluator.evaluate_multi_objective(chrom)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_invalid_chromosome_returns_penalty(self, evaluator):
        # Very short chromosome → repair will pad, should still work
        chrom = [0.5] * 3
        result = evaluator.evaluate(chrom)
        assert isinstance(result, tuple)


class TestFitnessCache:
    def test_put_and_get(self):
        cache = FitnessCache()
        chrom = [0.5] * 13
        cache.put(chrom, "iris", (0.95, 1.2))
        result = cache.get(chrom, "iris")
        assert result == (0.95, 1.2)

    def test_miss_returns_none(self):
        cache = FitnessCache()
        result = cache.get([0.5] * 13, "iris")
        assert result is None

    def test_clear(self):
        cache = FitnessCache()
        cache.put([0.5] * 13, "iris", (0.9,))
        cache.clear()
        assert cache.size == 0
