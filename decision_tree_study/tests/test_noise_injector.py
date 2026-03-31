"""Tests for data.noise_injector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.noise_injector import inject_feature_noise, inject_label_noise


# ── Label noise ──────────────────────────────────────────────────────


class TestInjectLabelNoise:
    def test_zero_noise_no_change(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_noisy, flipped, meta = inject_label_noise(y, 0.0, rng=rng)
        np.testing.assert_array_equal(y, y_noisy)
        assert flipped.sum() == 0
        assert meta["n_affected"] == 0

    def test_returns_copy(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 1, 2, 0, 1, 2])
        y_noisy, _, _ = inject_label_noise(y, 0.5, rng=rng)
        assert y_noisy is not y

    def test_symmetric_mode_flips(self):
        rng = np.random.default_rng(42)
        y = np.zeros(1000, dtype=int)
        y_noisy, flipped, meta = inject_label_noise(y, 0.3, mode="symmetric", rng=rng)
        # with only one class we can't actually flip, so test with 2 classes
        y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
        y_noisy, flipped, meta = inject_label_noise(y, 0.3, mode="symmetric", rng=rng)
        actual_rate = meta["n_affected"] / meta["n_total"]
        assert 0.15 < actual_rate < 0.45  # rough tolerance
        assert flipped.sum() > 0

    def test_asymmetric_mode(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 0, 1, 1, 2, 2] * 50)
        y_noisy, flipped, meta = inject_label_noise(y, 0.2, mode="asymmetric", rng=rng)
        # Asymmetric: class i → (i+1) mod C
        flipped_idxs = np.where(flipped)[0]
        for idx in flipped_idxs:
            expected = (y[idx] + 1) % 3
            assert y_noisy[idx] == expected

    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 1, 2, 3] * 25)
        y_noisy, flipped, _ = inject_label_noise(y, 0.1, rng=rng)
        assert y_noisy.shape == y.shape
        assert flipped.shape == y.shape


# ── Feature noise ────────────────────────────────────────────────────


class TestInjectFeatureNoise:
    def test_zero_sigma_no_change(self):
        rng = np.random.default_rng(42)
        X = np.ones((10, 3))
        X_noisy, meta = inject_feature_noise(X, sigma_factor=0.0, rng=rng)
        np.testing.assert_array_almost_equal(X, X_noisy)

    def test_returns_copy(self):
        rng = np.random.default_rng(42)
        X = np.ones((10, 3))
        X_noisy, _ = inject_feature_noise(X, sigma_factor=1.0, rng=rng)
        assert X_noisy is not X

    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        X = np.random.default_rng(0).standard_normal((50, 5))
        X_noisy, meta = inject_feature_noise(X, sigma_factor=0.5, rng=rng)
        assert X_noisy.shape == X.shape

    def test_noise_magnitude_scales(self):
        rng = np.random.default_rng(42)
        X = np.random.default_rng(0).standard_normal((200, 4))
        X_low, _ = inject_feature_noise(X, sigma_factor=0.1, rng=np.random.default_rng(42))
        X_high, _ = inject_feature_noise(X, sigma_factor=1.0, rng=np.random.default_rng(42))
        diff_low = np.abs(X - X_low).mean()
        diff_high = np.abs(X - X_high).mean()
        assert diff_high > diff_low

    def test_masking(self):
        rng = np.random.default_rng(42)
        X = np.ones((100, 5))
        X_noisy, meta = inject_feature_noise(X, sigma_factor=0.0, mask_rate=0.5, rng=rng)
        # About half the values should be zero
        zero_frac = (X_noisy == 0).mean()
        assert 0.3 < zero_frac < 0.7
