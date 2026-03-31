"""Tests for data.noise_injector."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.noise_injector import inject_label_noise, inject_feature_noise


class TestInjectLabelNoise:
    def test_zero_rate_returns_identical(self):
        y = np.array([0, 1, 0, 1, 2, 2])
        y_n, mask, meta = inject_label_noise(y, 0.0, np.random.default_rng(1))
        np.testing.assert_array_equal(y, y_n)
        assert mask.sum() == 0
        assert meta["n_affected"] == 0

    def test_returns_copy(self):
        y = np.array([0, 1, 0, 1])
        y_n, _, _ = inject_label_noise(y, 0.0, np.random.default_rng(1))
        assert y_n is not y

    def test_nonzero_rate_flips_labels(self):
        rng = np.random.default_rng(42)
        y = np.array([0] * 50 + [1] * 50)
        y_n, mask, meta = inject_label_noise(y, 0.2, rng)
        assert meta["n_affected"] > 0
        assert meta["rate"] == 0.2
        assert meta["type"] == "label_noise"

    def test_flipped_labels_differ_from_original(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 0, 1, 1, 2, 2] * 20)
        y_n, mask, _ = inject_label_noise(y, 0.3, rng)
        # All flipped positions should have a different label
        flipped_orig = y[mask]
        flipped_new = y_n[mask]
        assert np.all(flipped_orig != flipped_new)

    def test_reproducible_with_same_seed(self):
        y = np.array([0, 1, 2] * 30)
        y1, _, _ = inject_label_noise(y, 0.1, np.random.default_rng(99))
        y2, _, _ = inject_label_noise(y, 0.1, np.random.default_rng(99))
        np.testing.assert_array_equal(y1, y2)

    def test_meta_keys(self):
        y = np.array([0, 1] * 20)
        _, _, meta = inject_label_noise(y, 0.1, np.random.default_rng(1))
        assert set(meta.keys()) == {"type", "rate", "n_affected", "n_total"}


class TestInjectFeatureNoise:
    def test_zero_sigma_returns_identical(self):
        X = np.ones((10, 3))
        X_n, meta = inject_feature_noise(X, 0.0, np.random.default_rng(1))
        np.testing.assert_array_equal(X, X_n)
        assert meta["n_affected"] == 0

    def test_nonzero_sigma_changes_values(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 4))
        X_n, meta = inject_feature_noise(X, 0.5, np.random.default_rng(7))
        assert not np.allclose(X, X_n)
        assert meta["type"] == "feature_noise"

    def test_returns_copy(self):
        X = np.ones((5, 2))
        X_n, _ = inject_feature_noise(X, 0.0, np.random.default_rng(1))
        assert X_n is not X
