"""
XOR Example — The Classic Non-Linear Classification Demo
=========================================================

XOR cannot be solved by a single perceptron (Minsky & Papert, 1969).
A 2-layer network with a hidden layer of ≥ 2 neurons can learn it.

Truth table
-----------
::

    X1  X2  |  Y
    0   0   |  0
    0   1   |  1
    1   0   |  1
    1   1   |  0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── project imports ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Sigmoid
from src.core.layer import DenseLayer
from src.core.losses import BinaryCrossEntropyLoss
from src.core.optimizers import Adam
from src.network.sequential import Sequential
from src.network.model import Model
from src.utils.visualization import plot_training_curves


def main() -> None:
    np.random.seed(42)
    output_dir = ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ──
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[0],    [1],    [1],    [0]],     dtype=np.float64)

    # ── model ──
    network = Sequential(
        DenseLayer(2, 8, activation=ReLU(), seed=42),
        DenseLayer(8, 1, activation=Sigmoid(), seed=43),
    )

    model = Model(
        network=network,
        loss_fn=BinaryCrossEntropyLoss(),
        optimizer=Adam(lr=0.01),
    )

    print("=" * 50)
    print("XOR Example — Neural Network From Scratch")
    print("=" * 50)
    print(network.summary())
    print()

    # ── train ──
    history = model.fit(
        X, Y,
        epochs=1000,
        batch_size=4,
        verbose=False,
    )

    # ── evaluate ──
    Y_hat = model.predict(X)
    print("Predictions after training:")
    for i in range(len(X)):
        print(f"  {X[i]} → {Y_hat[i, 0]:.4f}  (target: {Y[i, 0]})")

    preds = (Y_hat >= 0.5).astype(int)
    acc = np.mean(preds == Y)
    print(f"\nAccuracy: {acc:.2%}")
    print(f"Final loss: {history['train_loss'][-1]:.6f}")

    # ── plot ──
    plot_training_curves(history, save_path=output_dir / "xor_training.png", title="XOR Training")
    print(f"Training curve saved to {output_dir / 'xor_training.png'}")


if __name__ == "__main__":
    main()
