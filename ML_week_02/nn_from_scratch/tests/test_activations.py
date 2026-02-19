"""
Tests for Activation Functions
==============================

Verifies forward pass outputs, backward pass gradients, and numerical
properties (shape preservation, value ranges, gradient checking).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax


# ── Helpers ──

def numerical_gradient(activation, Z, epsilon=1e-7):
    """Compute numerical gradient via centred differences."""
    grad = np.zeros_like(Z)
    it = np.nditer(Z, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = Z[idx]

        Z[idx] = orig + epsilon
        f_plus = activation.forward(Z).copy()

        Z[idx] = orig - epsilon
        f_minus = activation.forward(Z).copy()

        Z[idx] = orig
        grad[idx] = (f_plus[idx] - f_minus[idx]) / (2 * epsilon)
        it.iternext()

    # Restore forward cache
    activation.forward(Z)
    return grad


# ────────────────────────────────────────────────────────────────────
# ReLU
# ────────────────────────────────────────────────────────────────────
class TestReLU:
    def test_forward_positive(self):
        relu = ReLU()
        Z = np.array([[1.0, 2.0, 3.0]])
        A = relu.forward(Z)
        np.testing.assert_array_equal(A, Z)

    def test_forward_negative(self):
        relu = ReLU()
        Z = np.array([[-1.0, -2.0, -3.0]])
        A = relu.forward(Z)
        np.testing.assert_array_equal(A, np.zeros_like(Z))

    def test_forward_mixed(self):
        relu = ReLU()
        Z = np.array([[-1.0, 0.0, 2.0]])
        A = relu.forward(Z)
        expected = np.array([[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(A, expected)

    def test_backward_shape(self):
        relu = ReLU()
        Z = np.random.randn(5, 3)
        relu.forward(Z)
        dA = np.ones_like(Z)
        dZ = relu.backward(dA)
        assert dZ.shape == Z.shape

    def test_backward_numerical(self):
        relu = ReLU()
        Z = np.array([[0.5, -0.3, 1.2, -0.1]])
        relu.forward(Z)
        dA = np.ones_like(Z)
        dZ_analytic = relu.backward(dA)
        dZ_numeric = numerical_gradient(relu, Z.copy())
        np.testing.assert_allclose(dZ_analytic, dZ_numeric, atol=1e-5)


# ────────────────────────────────────────────────────────────────────
# LeakyReLU
# ────────────────────────────────────────────────────────────────────
class TestLeakyReLU:
    def test_forward(self):
        lrelu = LeakyReLU(alpha=0.01)
        Z = np.array([[-1.0, 0.0, 2.0]])
        A = lrelu.forward(Z)
        expected = np.array([[-0.01, 0.0, 2.0]])
        np.testing.assert_allclose(A, expected)

    def test_backward_numerical(self):
        lrelu = LeakyReLU(alpha=0.01)
        Z = np.array([[0.5, -0.3, 1.2, -0.8]])
        lrelu.forward(Z)
        dA = np.ones_like(Z)
        dZ_analytic = lrelu.backward(dA)
        dZ_numeric = numerical_gradient(lrelu, Z.copy())
        np.testing.assert_allclose(dZ_analytic, dZ_numeric, atol=1e-5)


# ────────────────────────────────────────────────────────────────────
# Sigmoid
# ────────────────────────────────────────────────────────────────────
class TestSigmoid:
    def test_forward_zero(self):
        sig = Sigmoid()
        A = sig.forward(np.array([[0.0]]))
        np.testing.assert_almost_equal(A[0, 0], 0.5)

    def test_forward_range(self):
        sig = Sigmoid()
        Z = np.random.randn(100, 10)
        A = sig.forward(Z)
        assert np.all(A >= 0.0) and np.all(A <= 1.0)

    def test_backward_numerical(self):
        sig = Sigmoid()
        Z = np.random.randn(3, 4)
        sig.forward(Z)
        dA = np.ones_like(Z)
        dZ_analytic = sig.backward(dA)
        dZ_numeric = numerical_gradient(sig, Z.copy())
        np.testing.assert_allclose(dZ_analytic, dZ_numeric, atol=1e-5)

    def test_backward_shape(self):
        sig = Sigmoid()
        Z = np.random.randn(8, 5)
        sig.forward(Z)
        dZ = sig.backward(np.ones_like(Z))
        assert dZ.shape == Z.shape


# ────────────────────────────────────────────────────────────────────
# Tanh
# ────────────────────────────────────────────────────────────────────
class TestTanh:
    def test_forward_range(self):
        t = Tanh()
        Z = np.random.randn(50, 10)
        A = t.forward(Z)
        assert np.all(A >= -1.0) and np.all(A <= 1.0)

    def test_backward_numerical(self):
        t = Tanh()
        Z = np.random.randn(3, 4)
        t.forward(Z)
        dA = np.ones_like(Z)
        dZ_analytic = t.backward(dA)
        dZ_numeric = numerical_gradient(t, Z.copy())
        np.testing.assert_allclose(dZ_analytic, dZ_numeric, atol=1e-5)


# ────────────────────────────────────────────────────────────────────
# Softmax
# ────────────────────────────────────────────────────────────────────
class TestSoftmax:
    def test_forward_sums_to_one(self):
        sm = Softmax()
        Z = np.random.randn(4, 5)
        A = sm.forward(Z)
        np.testing.assert_allclose(A.sum(axis=1), np.ones(4), atol=1e-10)

    def test_forward_positive(self):
        sm = Softmax()
        Z = np.random.randn(4, 5)
        A = sm.forward(Z)
        assert np.all(A > 0)

    def test_numerical_stability(self):
        sm = Softmax()
        Z = np.array([[1000.0, 1001.0, 1002.0]])
        A = sm.forward(Z)
        np.testing.assert_allclose(A.sum(), 1.0, atol=1e-10)
        assert np.all(np.isfinite(A))

    def test_backward_shape(self):
        sm = Softmax()
        Z = np.random.randn(4, 5)
        sm.forward(Z)
        dA = np.random.randn(4, 5)
        dZ = sm.backward(dA)
        assert dZ.shape == (4, 5)
