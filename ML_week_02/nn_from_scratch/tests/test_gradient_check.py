"""
Tests for Gradient Checking
============================

Validates the gradient verification utilities themselves,
plus end-to-end gradient checking on small networks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Sigmoid, Tanh
from src.core.layer import DenseLayer
from src.core.losses import CrossEntropyLoss, MSELoss
from src.network.sequential import Sequential
from src.utils.data_utils import one_hot_encode
from src.validation.gradient_check import gradient_check, gradient_check_layer


class TestGradientCheck:
    def test_quadratic_function(self):
        """f(x) = x², grad = 2x — should pass easily."""
        x = np.array([3.0, -2.0, 0.5])
        analytic = 2.0 * x
        rel_err = gradient_check(lambda p: float(np.sum(p ** 2)), x, analytic)
        assert rel_err < 1e-7

    def test_linear_function(self):
        """f(x) = sum(x), grad = ones — exact."""
        x = np.array([1.0, 2.0, 3.0])
        analytic = np.ones_like(x)
        rel_err = gradient_check(lambda p: float(np.sum(p)), x, analytic)
        assert rel_err < 1e-7

    def test_wrong_gradient_detected(self):
        """Intentionally wrong gradient should give high error."""
        x = np.array([3.0, -2.0, 0.5])
        wrong_grad = np.zeros_like(x)  # wrong!
        rel_err = gradient_check(lambda p: float(np.sum(p ** 2)), x, wrong_grad)
        assert rel_err > 0.5


class TestGradientCheckLayer:
    def test_dense_no_activation(self):
        layer = DenseLayer(3, 2, activation=None, seed=42)
        X = np.random.randn(4, 3)
        dY = np.random.randn(4, 2)
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-5, f"{key}: {err:.2e}"

    def test_dense_sigmoid(self):
        layer = DenseLayer(3, 2, activation=Sigmoid(), seed=42)
        X = np.random.randn(4, 3)
        dY = np.random.randn(4, 2)
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-5, f"{key}: {err:.2e}"

    def test_dense_tanh(self):
        layer = DenseLayer(3, 2, activation=Tanh(), seed=42)
        X = np.random.randn(4, 3)
        dY = np.random.randn(4, 2)
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-5, f"{key}: {err:.2e}"

    def test_dense_relu(self):
        """ReLU — avoid z≈0 region; use positive inputs."""
        rng = np.random.default_rng(42)
        layer = DenseLayer(3, 2, activation=ReLU(), seed=42)
        X = rng.uniform(0.2, 2.0, (4, 3))
        dY = rng.standard_normal((4, 2))
        errors = gradient_check_layer(layer, X, dY, verbose=False)
        for key, err in errors.items():
            assert err < 1e-4, f"{key}: {err:.2e}"


class TestEndToEndGradient:
    """Full forward-backward through a small network, then check per-layer."""

    def test_two_layer_network(self):
        rng = np.random.default_rng(42)
        net = Sequential(
            DenseLayer(3, 4, activation=Sigmoid(), seed=42),
            DenseLayer(4, 2, activation=None, seed=43),
        )
        X = rng.standard_normal((5, 3))
        Y = rng.standard_normal((5, 2))

        # Forward + loss
        Y_hat = net.forward(X)
        loss_fn = MSELoss()
        loss = loss_fn.forward(Y_hat, Y)

        # Backward
        dY = loss_fn.backward()
        net.backward(dY)

        # Check each layer individually
        for layer in net.layers:
            if layer.trainable:
                errors = gradient_check_layer(layer, layer._cache["X"],
                                              np.ones_like(layer.forward(layer._cache["X"])),
                                              verbose=False)
                for key, err in errors.items():
                    assert err < 1e-4, f"{repr(layer)} {key}: {err:.2e}"
