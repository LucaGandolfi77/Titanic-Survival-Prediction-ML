"""
Tests for DenseLayer — Forward / Backward / Shapes
====================================================

Validates the DenseLayer's linear transform, activation integration,
gradient shapes, and numerical gradient checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Sigmoid, Softmax
from src.core.layer import DenseLayer, Layer
from src.core.initializers import he_init, xavier_init, lecun_init, zeros_init
from src.validation.gradient_check import gradient_check_layer


# ────────────────────────────────────────────────────────────────────
# Layer base class
# ────────────────────────────────────────────────────────────────────
class TestLayerBase:
    def test_abstract_forward(self):
        layer = Layer()
        with pytest.raises(NotImplementedError):
            layer.forward(np.zeros((1, 1)))

    def test_abstract_backward(self):
        layer = Layer()
        with pytest.raises(NotImplementedError):
            layer.backward(np.zeros((1, 1)))


# ────────────────────────────────────────────────────────────────────
# DenseLayer construction
# ────────────────────────────────────────────────────────────────────
class TestDenseLayerInit:
    def test_weight_shape(self):
        layer = DenseLayer(10, 5, seed=0)
        assert layer.W.shape == (10, 5)
        assert layer.b.shape == (1, 5)

    def test_bias_zeros(self):
        layer = DenseLayer(10, 5, seed=0)
        np.testing.assert_array_equal(layer.b, np.zeros((1, 5)))

    def test_custom_initializer(self):
        layer = DenseLayer(10, 5, weight_init=xavier_init, seed=0)
        assert layer.W.shape == (10, 5)
        # Xavier should have smaller variance than He for same fan_in
        layer_he = DenseLayer(10, 5, weight_init=he_init, seed=0)
        # Both should be finite
        assert np.all(np.isfinite(layer.W))
        assert np.all(np.isfinite(layer_he.W))

    def test_repr(self):
        layer = DenseLayer(10, 5, activation=ReLU())
        assert "DenseLayer" in repr(layer)
        assert "ReLU" in repr(layer)


# ────────────────────────────────────────────────────────────────────
# Forward pass
# ────────────────────────────────────────────────────────────────────
class TestDenseForward:
    def test_output_shape(self):
        layer = DenseLayer(4, 3, seed=42)
        X = np.random.randn(8, 4)
        Y = layer.forward(X)
        assert Y.shape == (8, 3)

    def test_linear_no_activation(self):
        """Without activation, output = X @ W + b."""
        layer = DenseLayer(2, 3, activation=None, seed=42)
        X = np.array([[1.0, 2.0]])
        Y = layer.forward(X)
        expected = X @ layer.W + layer.b
        np.testing.assert_allclose(Y, expected)

    def test_with_relu(self):
        layer = DenseLayer(2, 3, activation=ReLU(), seed=42)
        X = np.random.randn(5, 2)
        Y = layer.forward(X)
        # ReLU output is non-negative
        assert np.all(Y >= 0)

    def test_with_sigmoid(self):
        layer = DenseLayer(2, 3, activation=Sigmoid(), seed=42)
        X = np.random.randn(5, 2)
        Y = layer.forward(X)
        assert np.all(Y >= 0) and np.all(Y <= 1)


# ────────────────────────────────────────────────────────────────────
# Backward pass
# ────────────────────────────────────────────────────────────────────
class TestDenseBackward:
    def test_gradient_shapes(self):
        layer = DenseLayer(4, 3, activation=ReLU(), seed=42)
        X = np.random.randn(8, 4)
        layer.forward(X)
        dY = np.random.randn(8, 3)
        dX = layer.backward(dY)
        assert dX.shape == (8, 4)
        assert layer.dW.shape == (4, 3)
        assert layer.db.shape == (1, 3)

    def test_no_activation_gradient(self):
        layer = DenseLayer(3, 2, activation=None, seed=42)
        X = np.random.randn(4, 3)
        layer.forward(X)
        dY = np.random.randn(4, 2)
        dX = layer.backward(dY)
        # dX should be dY @ W.T
        expected_dX = dY @ layer.W.T
        np.testing.assert_allclose(dX, expected_dX)

    def test_numerical_gradient_check_no_activation(self):
        """Numerical gradient check for a layer without activation."""
        layer = DenseLayer(3, 2, activation=None, seed=42)
        X = np.random.randn(4, 3)
        dY = np.random.randn(4, 2)
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-5, f"Gradient check failed for {key}: {err:.2e}"

    def test_numerical_gradient_check_relu(self):
        """Numerical gradient check for a layer with ReLU activation."""
        rng = np.random.default_rng(42)
        layer = DenseLayer(3, 2, activation=ReLU(), seed=42)
        # Avoid values near zero where ReLU is non-differentiable
        X = rng.uniform(0.1, 2.0, (4, 3))
        dY = rng.standard_normal((4, 2))
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-4, f"Gradient check failed for {key}: {err:.2e}"


# ────────────────────────────────────────────────────────────────────
# Properties
# ────────────────────────────────────────────────────────────────────
class TestDenseProperties:
    def test_params_dict(self):
        layer = DenseLayer(4, 3)
        p = layer.params
        assert "W" in p and "b" in p
        assert p["W"].shape == (4, 3)

    def test_grads_dict(self):
        layer = DenseLayer(4, 3)
        g = layer.grads
        assert "W" in g and "b" in g

    def test_trainable_flag(self):
        layer = DenseLayer(4, 3)
        assert layer.trainable is True


# ────────────────────────────────────────────────────────────────────
# Initializers
# ────────────────────────────────────────────────────────────────────
class TestInitializers:
    def test_he_shape(self):
        W = he_init(10, 5)
        assert W.shape == (10, 5)

    def test_xavier_shape(self):
        W = xavier_init(10, 5)
        assert W.shape == (10, 5)

    def test_lecun_shape(self):
        W = lecun_init(10, 5)
        assert W.shape == (10, 5)

    def test_zeros(self):
        W = zeros_init(3, 4)
        np.testing.assert_array_equal(W, np.zeros((3, 4)))

    def test_he_variance(self):
        """He init variance should be ≈ 2/fan_in."""
        rng = np.random.default_rng(42)
        W = he_init(1000, 500, rng=rng)
        actual_var = np.var(W)
        expected_var = 2.0 / 1000
        np.testing.assert_allclose(actual_var, expected_var, rtol=0.15)

    def test_xavier_variance(self):
        """Xavier uniform variance should be ≈ 2/(fan_in + fan_out) * (b-a)^2/12 = 6/(n_in+n_out) * 1/3."""
        rng = np.random.default_rng(42)
        W = xavier_init(1000, 500, rng=rng)
        actual_var = np.var(W)
        # Uniform[-limit, limit] has variance = limit^2 / 3 = 6/(n_in+n_out) / 3 = 2/(n_in+n_out)
        expected_var = 2.0 / (1000 + 500)
        np.testing.assert_allclose(actual_var, expected_var, rtol=0.15)
