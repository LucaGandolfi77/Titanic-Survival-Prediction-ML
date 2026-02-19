"""
Regression Example — Sine Wave Approximation
=============================================

Train a neural network to approximate f(x) = sin(x) from noisy samples.
Demonstrates that a network with sufficient hidden units can approximate
any continuous function (Universal Approximation Theorem).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU
from src.core.layer import DenseLayer
from src.core.losses import MSELoss
from src.core.optimizers import Adam
from src.network.sequential import Sequential
from src.network.model import Model
from src.utils.data_utils import train_test_split
from src.utils.visualization import plot_training_curves

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    np.random.seed(42)
    output_dir = ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ──
    n = 500
    rng = np.random.default_rng(42)
    X = rng.uniform(-2 * np.pi, 2 * np.pi, (n, 1))
    Y = np.sin(X) + rng.normal(0, 0.1, (n, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, seed=42)

    # ── model ──
    network = Sequential(
        DenseLayer(1, 64, activation=ReLU(), seed=42),
        DenseLayer(64, 64, activation=ReLU(), seed=43),
        DenseLayer(64, 1, activation=None, seed=44),   # linear output for regression
    )

    model = Model(
        network=network,
        loss_fn=MSELoss(),
        optimizer=Adam(lr=0.005),
    )

    print("=" * 50)
    print("Regression Example — sin(x) Approximation")
    print("=" * 50)
    print(network.summary())
    print()

    history = model.fit(
        X_train, Y_train,
        epochs=200,
        batch_size=32,
        X_val=X_test, Y_val=Y_test,
        verbose=True,
    )

    # ── evaluate ──
    Y_pred = model.predict(X_test)
    mse_val = np.mean((Y_pred - Y_test) ** 2)
    print(f"\nTest MSE: {mse_val:.6f}")

    # ── plot predictions ──
    X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
    Y_plot = model.predict(X_plot)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_test, Y_test, s=10, alpha=0.5, label="Test Data")
    ax.plot(X_plot, Y_plot, color="red", linewidth=2, label="NN Prediction")
    ax.plot(X_plot, np.sin(X_plot), "--", color="green", label="True sin(x)")
    ax.set_title("Regression: sin(x) Approximation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_fit.png", dpi=150)
    plt.close(fig)

    plot_training_curves(history, save_path=output_dir / "regression_training.png", title="Regression Training")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
