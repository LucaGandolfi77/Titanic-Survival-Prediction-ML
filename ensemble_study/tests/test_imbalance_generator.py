"""Tests for data.imbalance_generator."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.imbalance_generator import generate_imbalanced


class TestGenerateImbalanced:
    def setup_method(self):
        rng = np.random.default_rng(0)
        # 200 majority (class 0), 100 minority (class 1)
        self.X = rng.standard_normal((300, 4))
        self.y = np.array([0] * 200 + [1] * 100)

    def test_balanced_ratio_preserves_size(self):
        X_i, y_i, meta = generate_imbalanced(self.X, self.y, "1:1",
                                              np.random.default_rng(42))
        # ratio 1:1 → parse_ratio returns 1.0 → early return (copy)
        assert len(y_i) == len(self.y)
        assert meta["type"] == "imbalance"

    def test_imbalanced_reduces_majority(self):
        X_i, y_i, meta = generate_imbalanced(self.X, self.y, "1:5",
                                              np.random.default_rng(42))
        # With 100 minority and ratio 1:5 → target_maj = 100/0.2 = 500
        # but original majority is only 200, so it keeps all 200
        assert len(y_i) <= len(self.y)
        assert meta["n_samples_after"] <= meta["n_samples_before"]

    def test_returns_copies(self):
        X_i, y_i, _ = generate_imbalanced(self.X, self.y, "1:2",
                                           np.random.default_rng(42))
        assert X_i is not self.X
        assert y_i is not self.y

    def test_meta_keys(self):
        _, _, meta = generate_imbalanced(self.X, self.y, "1:2",
                                          np.random.default_rng(42))
        assert set(meta.keys()) == {"type", "ratio", "n_samples_after", "n_samples_before"}

    def test_reproducibility(self):
        X1, y1, _ = generate_imbalanced(self.X, self.y, "1:5",
                                         np.random.default_rng(42))
        X2, y2, _ = generate_imbalanced(self.X, self.y, "1:5",
                                         np.random.default_rng(42))
        np.testing.assert_array_equal(y1, y2)

    def test_extreme_ratio(self):
        X_i, y_i, _ = generate_imbalanced(self.X, self.y, "1:20",
                                           np.random.default_rng(42))
        classes, counts = np.unique(y_i, return_counts=True)
        minority_count = counts[classes == 1][0]
        majority_count = counts[classes == 0][0]
        assert majority_count >= minority_count
