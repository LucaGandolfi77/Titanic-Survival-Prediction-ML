"""
High-Level Model Interface
==========================

``Model`` wraps a ``Sequential`` network, a ``Loss``, and an ``Optimizer``
to provide a convenient training loop with:

  • ``fit(X, Y, ...)``   — full training with epochs, batches, early-stop
  • ``evaluate(X, Y)``   — compute loss + accuracy on a dataset
  • ``predict(X)``       — forward-only inference

This mirrors the Keras-style API while every computation under the hood
uses only NumPy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core.losses import Loss
from ..core.optimizers import Optimizer
from ..utils.data_utils import BatchGenerator, shuffle_data
from .sequential import Sequential


class Model:
    """High-level wrapper around Sequential + Loss + Optimizer.

    Parameters
    ----------
    network   : Sequential — the layer stack.
    loss_fn   : Loss — loss function instance.
    optimizer : Optimizer — parameter update rule.
    """

    def __init__(
        self,
        network: Sequential,
        loss_fn: Loss,
        optimizer: Optimizer,
    ) -> None:
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    # ── training loop ────────────────────────────────────────────
    def fit(
        self,
        X_train: NDArray,
        Y_train: NDArray,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: NDArray | None = None,
        Y_val: NDArray | None = None,
        verbose: bool = True,
        early_stop_patience: int = 0,
        shuffle: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
        Y_train : ndarray, shape (n_train, n_classes) or (n_train, 1)
        epochs  : int — number of full passes over the data.
        batch_size : int — mini-batch size.
        X_val, Y_val : optional validation set.
        verbose : bool — print epoch metrics.
        early_stop_patience : int — stop if val_loss doesn't improve for
            this many epochs (0 = disabled).
        shuffle : bool — shuffle training data each epoch.

        Returns
        -------
        history : dict — lists of per-epoch metrics.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── shuffle ──
            if shuffle:
                X_train, Y_train = shuffle_data(X_train, Y_train)

            # ── mini-batch training ──
            epoch_loss = 0.0
            n_batches = 0
            batch_gen = BatchGenerator(X_train, Y_train, batch_size)

            for X_batch, Y_batch in batch_gen:
                # Forward
                Y_hat = self.network.forward(X_batch)
                loss = self.loss_fn.forward(Y_hat, Y_batch)

                # Backward
                dY = self.loss_fn.backward()
                self.network.backward(dY)

                # Update parameters
                self.optimizer.step(self.network.layers)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_acc = self._accuracy(X_train, Y_train)
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(train_acc)

            # ── validation ──
            if X_val is not None and Y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, Y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
            else:
                val_loss = None
                val_acc = None

            elapsed = time.time() - t0

            if verbose:
                msg = (
                    f"Epoch {epoch:>4d}/{epochs} — "
                    f"loss: {avg_loss:.4f}  acc: {train_acc:.4f}"
                )
                if val_loss is not None:
                    msg += f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
                msg += f"  ({elapsed:.2f}s)"
                print(msg)

            # ── early stopping ──
            if early_stop_patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(val_loss didn't improve for "
                                f"{early_stop_patience} epochs)"
                            )
                        break

        return self.history

    # ── evaluation ───────────────────────────────────────────────
    def evaluate(self, X: NDArray, Y: NDArray) -> tuple[float, float]:
        """Compute loss and accuracy on a dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        Y : ndarray, shape (n_samples, n_classes) or (n_samples, 1)

        Returns
        -------
        (loss, accuracy) : tuple[float, float]
        """
        Y_hat = self.network.forward(X)
        loss = self.loss_fn.forward(Y_hat, Y)
        acc = self._accuracy(X, Y)
        return loss, acc

    # ── prediction ───────────────────────────────────────────────
    def predict(self, X: NDArray) -> NDArray:
        """Forward-only inference.

        Returns
        -------
        Y_hat : ndarray
        """
        return self.network.forward(X)

    # ── save / load ──────────────────────────────────────────────
    def save_weights(self, path: Path | str) -> None:
        """Save all trainable parameters to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, NDArray] = {}
        for i, layer in enumerate(self.network.layers):
            if layer.trainable:
                for key, val in layer.params.items():
                    data[f"layer_{i}_{key}"] = val
        np.savez(path, **data)

    def load_weights(self, path: Path | str) -> None:
        """Load parameters from a .npz file."""
        path = Path(path)
        data = np.load(path)
        for i, layer in enumerate(self.network.layers):
            if layer.trainable:
                for key in layer.params:
                    arr_key = f"layer_{i}_{key}"
                    if arr_key in data:
                        layer.params[key][...] = data[arr_key]

    # ── internal helpers ─────────────────────────────────────────
    def _accuracy(self, X: NDArray, Y: NDArray) -> float:
        """Compute classification accuracy (argmax-based)."""
        Y_hat = self.network.forward(X)
        if Y_hat.shape[1] == 1:
            # binary — threshold at 0.5
            preds = (Y_hat >= 0.5).astype(int)
            labels = Y.astype(int)
        else:
            # multi-class — argmax
            preds = np.argmax(Y_hat, axis=1)
            labels = np.argmax(Y, axis=1) if Y.ndim > 1 and Y.shape[1] > 1 else Y.ravel()
        return float(np.mean(preds.ravel() == labels.ravel()))

    def __repr__(self) -> str:
        return (
            f"Model(\n"
            f"  network={self.network},\n"
            f"  loss={self.loss_fn},\n"
            f"  optimizer={self.optimizer}\n"
            f")"
        )
