"""
Tests for Sequential Network and Model
=======================================

Integration tests covering:
  • Sequential forward / backward
  • Model training on XOR
  • Save / load weights
  • Optimizers (SGD, Momentum, Adam) convergence on toy data
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Sigmoid, Softmax, Tanh
from src.core.layer import DenseLayer
from src.core.losses import CrossEntropyLoss, MSELoss, BinaryCrossEntropyLoss
from src.core.optimizers import SGD, Momentum, Adam
from src.network.sequential import Sequential
from src.network.model import Model
from src.utils.data_utils import one_hot_encode, train_test_split, shuffle_data, BatchGenerator
from src.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix


# ────────────────────────────────────────────────────────────────────
# Sequential
# ────────────────────────────────────────────────────────────────────
class TestSequential:
    def test_forward_shape(self):
        net = Sequential(
            DenseLayer(4, 8, activation=ReLU(), seed=0),
            DenseLayer(8, 3, activation=Softmax(), seed=1),
        )
        X = np.random.randn(16, 4)
        Y = net.forward(X)
        assert Y.shape == (16, 3)

    def test_softmax_sums_to_one(self):
        net = Sequential(
            DenseLayer(4, 3, activation=Softmax(), seed=0),
        )
        X = np.random.randn(8, 4)
        Y = net.forward(X)
        np.testing.assert_allclose(Y.sum(axis=1), np.ones(8), atol=1e-10)

    def test_backward_shape(self):
        net = Sequential(
            DenseLayer(4, 8, activation=ReLU(), seed=0),
            DenseLayer(8, 3, seed=1),
        )
        X = np.random.randn(16, 4)
        net.forward(X)
        dY = np.random.randn(16, 3)
        dX = net.backward(dY)
        assert dX.shape == (16, 4)

    def test_add_layer(self):
        net = Sequential()
        net.add(DenseLayer(4, 8))
        net.add(DenseLayer(8, 2))
        assert len(net) == 2

    def test_count_params(self):
        net = Sequential(
            DenseLayer(10, 5, seed=0),   # 10*5 + 5 = 55
            DenseLayer(5, 3, seed=1),    # 5*3 + 3 = 18
        )
        assert net.count_params() == 55 + 18

    def test_summary_string(self):
        net = Sequential(DenseLayer(4, 3, activation=ReLU()))
        s = net.summary()
        assert "Total trainable params" in s

    def test_predict_same_as_forward(self):
        net = Sequential(DenseLayer(4, 3, seed=0))
        X = np.random.randn(5, 4)
        np.testing.assert_array_equal(net.forward(X), net.predict(X))


# ────────────────────────────────────────────────────────────────────
# Model — XOR convergence
# ────────────────────────────────────────────────────────────────────
class TestModelXOR:
    @pytest.fixture
    def xor_data(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        Y = np.array([[0],    [1],    [1],    [0]],     dtype=np.float64)
        return X, Y

    def test_train_converges_adam(self, xor_data):
        X, Y = xor_data
        net = Sequential(
            DenseLayer(2, 16, activation=ReLU(), seed=42),
            DenseLayer(16, 1, activation=Sigmoid(), seed=43),
        )
        model = Model(net, BinaryCrossEntropyLoss(), Adam(lr=0.01))
        history = model.fit(X, Y, epochs=500, batch_size=4, verbose=False)
        assert history["train_loss"][-1] < 0.1

    def test_train_converges_sgd(self, xor_data):
        X, Y = xor_data
        net = Sequential(
            DenseLayer(2, 16, activation=ReLU(), seed=42),
            DenseLayer(16, 1, activation=Sigmoid(), seed=43),
        )
        model = Model(net, BinaryCrossEntropyLoss(), SGD(lr=0.5))
        history = model.fit(X, Y, epochs=1000, batch_size=4, verbose=False)
        assert history["train_loss"][-1] < 0.5

    def test_train_converges_momentum(self, xor_data):
        X, Y = xor_data
        net = Sequential(
            DenseLayer(2, 16, activation=ReLU(), seed=42),
            DenseLayer(16, 1, activation=Sigmoid(), seed=43),
        )
        model = Model(net, BinaryCrossEntropyLoss(), Momentum(lr=0.5, beta=0.9))
        history = model.fit(X, Y, epochs=1000, batch_size=4, verbose=False)
        assert history["train_loss"][-1] < 0.5

    def test_predict_shape(self, xor_data):
        X, Y = xor_data
        net = Sequential(
            DenseLayer(2, 4, activation=ReLU(), seed=0),
            DenseLayer(4, 1, activation=Sigmoid(), seed=1),
        )
        model = Model(net, BinaryCrossEntropyLoss(), Adam())
        Y_hat = model.predict(X)
        assert Y_hat.shape == (4, 1)


# ────────────────────────────────────────────────────────────────────
# Model — multi-class
# ────────────────────────────────────────────────────────────────────
class TestModelMultiClass:
    def test_softmax_cross_entropy(self):
        rng = np.random.default_rng(42)
        n, d, k = 100, 4, 3
        X = rng.standard_normal((n, d))
        Y_int = rng.integers(0, k, n)
        Y = one_hot_encode(Y_int, k)

        net = Sequential(
            DenseLayer(d, 32, activation=ReLU(), seed=42),
            DenseLayer(32, k, activation=Softmax(), seed=43),
        )
        model = Model(net, CrossEntropyLoss(), Adam(lr=0.01))
        history = model.fit(X, Y, epochs=100, batch_size=32, verbose=False)
        # Should at least beat random (33%)
        preds = np.argmax(model.predict(X), axis=1)
        acc = np.mean(preds == Y_int)
        assert acc > 0.4


# ────────────────────────────────────────────────────────────────────
# Model — regression
# ────────────────────────────────────────────────────────────────────
class TestModelRegression:
    def test_mse_convergence(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (100, 1))
        Y = 2 * X + 1 + rng.normal(0, 0.1, (100, 1))

        net = Sequential(
            DenseLayer(1, 16, activation=ReLU(), seed=42),
            DenseLayer(16, 1, activation=None, seed=43),
        )
        model = Model(net, MSELoss(), Adam(lr=0.01))
        history = model.fit(X, Y, epochs=200, batch_size=32, verbose=False)
        assert history["train_loss"][-1] < 0.5


# ────────────────────────────────────────────────────────────────────
# Save / Load
# ────────────────────────────────────────────────────────────────────
class TestSaveLoad:
    def test_roundtrip(self):
        net = Sequential(
            DenseLayer(4, 8, activation=ReLU(), seed=42),
            DenseLayer(8, 3, activation=Softmax(), seed=43),
        )
        model = Model(net, CrossEntropyLoss(), Adam())
        X = np.random.randn(5, 4)
        Y_before = model.predict(X)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "weights.npz"
            model.save_weights(p)
            assert p.exists()

            # Create fresh model with same architecture
            net2 = Sequential(
                DenseLayer(4, 8, activation=ReLU(), seed=99),
                DenseLayer(8, 3, activation=Softmax(), seed=99),
            )
            model2 = Model(net2, CrossEntropyLoss(), Adam())
            model2.load_weights(p)
            Y_after = model2.predict(X)
            np.testing.assert_allclose(Y_before, Y_after, atol=1e-10)


# ────────────────────────────────────────────────────────────────────
# Data utilities
# ────────────────────────────────────────────────────────────────────
class TestDataUtils:
    def test_one_hot_encode(self):
        labels = np.array([0, 1, 2, 1])
        oh = one_hot_encode(labels, 3)
        assert oh.shape == (4, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_equal(oh, expected)

    def test_train_test_split_sizes(self):
        X = np.arange(100).reshape(100, 1)
        Y = np.arange(100)
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, seed=42)
        assert Xtr.shape[0] == 80
        assert Xte.shape[0] == 20

    def test_shuffle_preserves_data(self):
        X = np.arange(10).reshape(10, 1).astype(float)
        Y = np.arange(10).astype(float)
        rng = np.random.default_rng(42)
        Xs, Ys = shuffle_data(X, Y, rng)
        assert set(Xs.ravel()) == set(X.ravel())
        assert set(Ys.ravel()) == set(Y.ravel())

    def test_batch_generator(self):
        X = np.arange(10).reshape(10, 1).astype(float)
        Y = np.arange(10).astype(float)
        batches = list(BatchGenerator(X, Y, batch_size=3))
        assert len(batches) == 4  # 3+3+3+1
        total = sum(b[0].shape[0] for b in batches)
        assert total == 10


# ────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_accuracy_perfect(self):
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 2, 1])
        assert accuracy(y_true, y_pred) == 1.0

    def test_accuracy_half(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert accuracy(y_true, y_pred) == 0.5

    def test_precision_and_recall(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        assert 0 <= p <= 1
        assert 0 <= r <= 1

    def test_f1_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert f1_score(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 2, 2, 1, 1])
        cm = confusion_matrix(y_true, y_pred, n_classes=3)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(np.diag(cm), [2, 2, 2])


# ────────────────────────────────────────────────────────────────────
# Early stopping
# ────────────────────────────────────────────────────────────────────
class TestEarlyStopping:
    def test_early_stop_triggers(self):
        """Model should stop before max epochs when validation doesn't improve."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        Y = np.zeros((20, 1))
        # Val set with different distribution to ensure val_loss stalls
        X_val = rng.standard_normal((10, 2)) * 100
        Y_val = np.ones((10, 1))

        net = Sequential(
            DenseLayer(2, 4, activation=Sigmoid(), seed=42),
            DenseLayer(4, 1, activation=Sigmoid(), seed=43),
        )
        model = Model(net, BinaryCrossEntropyLoss(), Adam(lr=0.001))
        history = model.fit(
            X, Y,
            epochs=500,
            batch_size=20,
            X_val=X_val, Y_val=Y_val,
            early_stop_patience=10,
            verbose=False,
        )
        # Should have stopped well before 500 epochs
        assert len(history["train_loss"]) < 500
