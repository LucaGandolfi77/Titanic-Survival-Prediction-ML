"""
Tests for Loss Functions
========================

Validates loss values, gradient shapes, and numerical gradient checking
for CrossEntropy, MSE, and BinaryCrossEntropy losses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.losses import CrossEntropyLoss, MSELoss, BinaryCrossEntropyLoss


# ────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss
# ────────────────────────────────────────────────────────────────────
class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        """Loss ≈ 0 when prediction matches target perfectly."""
        ce = CrossEntropyLoss()
        Y_hat = np.array([[0.99, 0.005, 0.005]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = ce.forward(Y_hat, Y)
        assert loss < 0.02

    def test_worst_prediction(self):
        """Loss should be high when prediction is opposite of target."""
        ce = CrossEntropyLoss()
        Y_hat = np.array([[0.01, 0.01, 0.98]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = ce.forward(Y_hat, Y)
        assert loss > 2.0

    def test_loss_non_negative(self):
        ce = CrossEntropyLoss()
        rng = np.random.default_rng(42)
        n, k = 16, 5
        logits = rng.standard_normal((n, k))
        # softmax to make valid probabilities
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        Y_hat = exp_l / exp_l.sum(axis=1, keepdims=True)
        # random one-hot
        labels = rng.integers(0, k, n)
        Y = np.zeros((n, k))
        Y[np.arange(n), labels] = 1.0
        loss = ce.forward(Y_hat, Y)
        assert loss >= 0.0

    def test_backward_shape(self):
        ce = CrossEntropyLoss()
        Y_hat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        Y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ce.forward(Y_hat, Y)
        dZ = ce.backward()
        assert dZ.shape == (2, 3)

    def test_backward_combined_gradient(self):
        """dZ = Y_hat - Y (softmax + CE combined)."""
        ce = CrossEntropyLoss()
        Y_hat = np.array([[0.7, 0.2, 0.1]])
        Y = np.array([[1.0, 0.0, 0.0]])
        ce.forward(Y_hat, Y)
        dZ = ce.backward()
        expected = Y_hat - Y
        np.testing.assert_allclose(dZ, expected)


# ────────────────────────────────────────────────────────────────────
# MSE Loss
# ────────────────────────────────────────────────────────────────────
class TestMSELoss:
    def test_zero_loss(self):
        mse = MSELoss()
        Y = np.array([[1.0], [2.0], [3.0]])
        loss = mse.forward(Y, Y)
        np.testing.assert_almost_equal(loss, 0.0)

    def test_known_value(self):
        mse = MSELoss()
        Y_hat = np.array([[1.0], [2.0]])
        Y = np.array([[0.0], [0.0]])
        # MSE = (1/2)(1² + 4) = 5/2 = 2.5
        loss = mse.forward(Y_hat, Y)
        np.testing.assert_almost_equal(loss, 2.5)

    def test_backward_shape(self):
        mse = MSELoss()
        Y_hat = np.random.randn(8, 1)
        Y = np.random.randn(8, 1)
        mse.forward(Y_hat, Y)
        dY = mse.backward()
        assert dY.shape == (8, 1)

    def test_backward_numerical(self):
        """Numerically verify MSE gradient."""
        mse = MSELoss()
        rng = np.random.default_rng(42)
        Y_hat = rng.standard_normal((4, 2))
        Y = rng.standard_normal((4, 2))
        mse.forward(Y_hat, Y)
        dY_analytic = mse.backward()

        # Numerical gradient
        eps = 1e-7
        dY_numeric = np.zeros_like(Y_hat)
        it = np.nditer(Y_hat, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = Y_hat[idx]
            Y_hat[idx] = orig + eps
            loss_p = mse.forward(Y_hat, Y)
            Y_hat[idx] = orig - eps
            loss_m = mse.forward(Y_hat, Y)
            Y_hat[idx] = orig
            dY_numeric[idx] = (loss_p - loss_m) / (2 * eps)
            it.iternext()

        np.testing.assert_allclose(dY_analytic, dY_numeric, atol=1e-5)


# ────────────────────────────────────────────────────────────────────
# Binary Cross-Entropy Loss
# ────────────────────────────────────────────────────────────────────
class TestBinaryCrossEntropyLoss:
    def test_perfect_prediction(self):
        bce = BinaryCrossEntropyLoss()
        Y_hat = np.array([[0.99], [0.01]])
        Y = np.array([[1.0], [0.0]])
        loss = bce.forward(Y_hat, Y)
        assert loss < 0.02

    def test_backward_shape(self):
        bce = BinaryCrossEntropyLoss()
        Y_hat = np.array([[0.7], [0.3]])
        Y = np.array([[1.0], [0.0]])
        bce.forward(Y_hat, Y)
        dY = bce.backward()
        assert dY.shape == (2, 1)

    def test_backward_numerical(self):
        bce = BinaryCrossEntropyLoss()
        rng = np.random.default_rng(42)
        Y_hat = rng.uniform(0.1, 0.9, (4, 1))
        Y = rng.integers(0, 2, (4, 1)).astype(float)
        bce.forward(Y_hat, Y)
        dY_analytic = bce.backward()

        eps = 1e-7
        dY_numeric = np.zeros_like(Y_hat)
        it = np.nditer(Y_hat, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = Y_hat[idx]
            Y_hat[idx] = orig + eps
            loss_p = bce.forward(Y_hat, Y)
            Y_hat[idx] = orig - eps
            loss_m = bce.forward(Y_hat, Y)
            Y_hat[idx] = orig
            dY_numeric[idx] = (loss_p - loss_m) / (2 * eps)
            it.iternext()

        np.testing.assert_allclose(dY_analytic, dY_numeric, atol=1e-5)


# ────────────────────────────────────────────────────────────────────
# Edge cases
# ────────────────────────────────────────────────────────────────────
class TestLossEdgeCases:
    def test_ce_single_sample(self):
        ce = CrossEntropyLoss()
        Y_hat = np.array([[0.5, 0.3, 0.2]])
        Y = np.array([[0.0, 1.0, 0.0]])
        loss = ce.forward(Y_hat, Y)
        assert np.isfinite(loss)

    def test_mse_single_sample(self):
        mse = MSELoss()
        Y_hat = np.array([[2.5]])
        Y = np.array([[3.0]])
        loss = mse.forward(Y_hat, Y)
        np.testing.assert_almost_equal(loss, 0.25)
