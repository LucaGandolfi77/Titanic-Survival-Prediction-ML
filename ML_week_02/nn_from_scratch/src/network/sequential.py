"""
Sequential Container
====================

A ``Sequential`` model chains layers in order, forwarding the output of
each layer as input to the next.  Backward propagation flows in reverse.

Architecture diagram
--------------------
::

    X ─→ [Layer 0] ─→ [Layer 1] ─→ ... ─→ [Layer N-1] ─→ Ŷ

    ∂L/∂X ←─ [Layer 0] ←─ [Layer 1] ←─ ... ←─ [Layer N-1] ←─ ∂L/∂Ŷ
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ..core.layer import Layer


class Sequential:
    """Sequential container — holds an ordered list of layers.

    Parameters
    ----------
    *layers : Layer
        Variadic initial layers (can also be added later with ``add``).

    Example
    -------
    >>> from src.core import DenseLayer, ReLU, Softmax
    >>> net = Sequential(
    ...     DenseLayer(784, 128, activation=ReLU()),
    ...     DenseLayer(128, 10,  activation=Softmax()),
    ... )
    >>> Y_hat = net.forward(X_batch)
    """

    def __init__(self, *layers: Layer) -> None:
        self._layers: list[Layer] = list(layers)

    # ── layer management ─────────────────────────────────────────
    def add(self, layer: Layer) -> "Sequential":
        """Append a layer and return self (for chaining)."""
        self._layers.append(layer)
        return self

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    @property
    def trainable_layers(self) -> list[Layer]:
        return [l for l in self._layers if l.trainable]

    # ── forward ──────────────────────────────────────────────────
    def forward(self, X: NDArray) -> NDArray:
        """Forward pass: pipe X through every layer.

        Parameters
        ----------
        X : ndarray, shape (batch_size, n_features)

        Returns
        -------
        Y_hat : ndarray — output of the last layer.
        """
        out: NDArray = X
        for layer in self._layers:
            out = layer.forward(out)
        return out

    # ── backward ─────────────────────────────────────────────────
    def backward(self, dY: NDArray) -> NDArray:
        """Backward pass: propagate gradient in reverse layer order.

        Parameters
        ----------
        dY : ndarray — gradient from the loss function.

        Returns
        -------
        dX : ndarray — gradient w.r.t. the network input (rarely used).
        """
        grad: NDArray = dY
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    # ── predict (inference shortcut) ─────────────────────────────
    def predict(self, X: NDArray) -> NDArray:
        """Alias for ``forward`` — semantically used at inference time."""
        return self.forward(X)

    # ── utilities ────────────────────────────────────────────────
    def count_params(self) -> int:
        """Total number of trainable scalar parameters."""
        total = 0
        for layer in self._layers:
            for p in layer.params.values():
                total += p.size
        return total

    def summary(self) -> str:
        """Print a Keras-style model summary."""
        lines: list[str] = []
        header = f"{'Layer':<30} {'Output Shape':<20} {'# Params':>10}"
        lines.append(header)
        lines.append("=" * len(header))
        total = 0
        for i, layer in enumerate(self._layers):
            n_params = sum(p.size for p in layer.params.values())
            total += n_params
            out_shape = ""
            if hasattr(layer, "n_out"):
                out_shape = f"(batch, {layer.n_out})"  # type: ignore[attr-defined]
            lines.append(f"{str(layer):<30} {out_shape:<20} {n_params:>10,}")
        lines.append("=" * len(header))
        lines.append(f"Total trainable params: {total:,}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(l) for l in self._layers)
        return f"Sequential(\n  {inner}\n)"
