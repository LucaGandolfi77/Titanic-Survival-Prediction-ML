"""Tests for genetic operators: crossover and mutation."""
from __future__ import annotations

import random

import pytest

from evolutionary_automl.genome.operators import cx_two_point_typed, mut_mixed_type


class TestCrossover:
    def test_preserves_length(self):
        random.seed(42)
        a = [random.random() for _ in range(13)]
        b = [random.random() for _ in range(13)]
        cx_two_point_typed(a, b)
        assert len(a) == 13
        assert len(b) == 13

    def test_output_in_bounds(self):
        random.seed(42)
        for _ in range(100):
            a = [random.random() for _ in range(13)]
            b = [random.random() for _ in range(13)]
            cx_two_point_typed(a, b)
            for g in a + b:
                assert 0.0 <= g <= 1.0

    def test_produces_different_offspring(self):
        random.seed(42)
        a = [0.5] * 13
        b = [0.1] * 13
        a_orig = list(a)
        b_orig = list(b)
        cx_two_point_typed(a, b)
        # At least one should have changed
        assert a != a_orig or b != b_orig


class TestMutation:
    def test_preserves_length(self):
        random.seed(42)
        ind = [random.random() for _ in range(13)]
        mut_mixed_type(ind, indpb=0.5)
        assert len(ind) == 13

    def test_output_in_bounds(self):
        random.seed(42)
        for _ in range(100):
            ind = [random.random() for _ in range(13)]
            mut_mixed_type(ind, indpb=0.5)
            for g in ind:
                assert 0.0 <= g <= 1.0

    def test_returns_tuple(self):
        random.seed(42)
        ind = [0.5] * 13
        result = mut_mixed_type(ind, indpb=0.5)
        assert isinstance(result, tuple)
        assert result[0] is ind

    def test_high_indpb_changes_individual(self):
        random.seed(42)
        ind = [0.5] * 13
        original = list(ind)
        mut_mixed_type(ind, indpb=1.0)
        assert ind != original
