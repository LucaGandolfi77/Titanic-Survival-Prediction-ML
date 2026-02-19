"""
Spiral Classification Example
==============================

A synthetic 2D spiral dataset with K classes. Points spiral outward
from the origin — a highly non-linear problem requiring multiple
hidden layers.

Dataset generation
------------------
For class k ∈ {0, …, K−1}:

.. math::
    r &= \\text{linspace}(0, 1, N) \\\\
    \\theta &= \\frac{4\\pi k}{K} + 4\\pi r + \\mathcal{N}(0, 0.25)  \\\\
    x_1 &= r \\sin(\\theta), \\quad x_2 = r \\cos(\\theta)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Softmax
from src.core.layer import DenseLayer
from src.core.losses import CrossEntropyLoss
from src.core.optimizers import Adam
from src.network.sequential import Sequential
from src.network.model import Model
from src.utils.data_utils import one_hot_encode
from src.utils.visualization import plot_training_curves, plot_decision_boundary


def make_spiral(n_samples: int = 100, n_classes: int = 3, seed: int = 42):
    """Generate spiral dataset.

    Returns
    -------
    X : ndarray, shape (n_samples * n_classes, 2)
    Y : ndarray, shape (n_samples * n_classes,) — integer labels.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples * n_classes, 2))
    Y = np.zeros(n_samples * n_classes, dtype=int)

    for k in range(n_classes):
        idx = range(n_samples * k, n_samples * (k + 1))
        r = np.linspace(0.0, 1.0, n_samples)
        theta = (4.0 * np.pi * k / n_classes) + 4.0 * np.pi * r + rng.normal(0, 0.25, n_samples)
        X[idx] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        Y[idx] = k

    return X, Y


def main() -> None:
    output_dir = ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 3
    X, Y_int = make_spiral(n_samples=150, n_classes=n_classes)
    Y_oh = one_hot_encode(Y_int, n_classes)

    # ── model ──
    network = Sequential(
        DenseLayer(2, 64, activation=ReLU(), seed=42),
        DenseLayer(64, 64, activation=ReLU(), seed=43),
        DenseLayer(64, n_classes, activation=Softmax(), seed=44),
    )

    model = Model(
        network=network,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(lr=0.01),
    )

    print("=" * 50)
    print("Spiral Classification — Neural Network From Scratch")
    print("=" * 50)
    print(network.summary())
    print()

    history = model.fit(X, Y_oh, epochs=300, batch_size=64, verbose=True)

    # ── accuracy ──
    Y_hat = model.predict(X)
    preds = np.argmax(Y_hat, axis=1)
    acc = np.mean(preds == Y_int)
    print(f"\nFinal Accuracy: {acc:.2%}")

    # ── plots ──
    plot_training_curves(history, save_path=output_dir / "spiral_training.png", title="Spiral Training")

    def predict_classes(X_grid):
        probs = model.predict(X_grid)
        return np.argmax(probs, axis=1)

    plot_decision_boundary(
        predict_classes, X, Y_int,
        save_path=output_dir / "spiral_boundary.png",
        title="Spiral Decision Boundary",
    )
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
