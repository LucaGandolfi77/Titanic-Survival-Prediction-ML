"""
Layer Abstractions — Base Layer & Dense (Fully-Connected) Layer
===============================================================

A *layer* transforms an input tensor X into an output tensor Y, caches
the quantities needed for the backward pass, and computes gradients.

Notation
--------
  X  : input          — shape (batch_size, n_in)
  Y  : output         — shape (batch_size, n_out)
  W  : weight matrix  — shape (n_in, n_out)
  b  : bias vector    — shape (1, n_out)
  dY : upstream grad  — ∂L/∂Y, shape (batch_size, n_out)
  dX : downstream grad — ∂L/∂X, shape (batch_size, n_in)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .activations import Activation
from .initializers import he_init, zeros_init


# ────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────
class Layer:
    """Abstract layer interface.

    Every concrete layer must implement ``forward`` and ``backward``.
    """

    trainable: bool = False

    def forward(self, X: NDArray) -> NDArray:  # noqa: N803
        raise NotImplementedError

    def backward(self, dY: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def params(self) -> dict[str, NDArray]:
        """Return dict of trainable parameters."""
        return {}

    @property
    def grads(self) -> dict[str, NDArray]:
        """Return dict of parameter gradients (same keys as ``params``)."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ────────────────────────────────────────────────────────────────────
# Dense (fully-connected) layer
# ────────────────────────────────────────────────────────────────────
class DenseLayer(Layer):
    r"""Fully-connected (dense / linear) layer with optional activation.

    Forward pass
    -------------
    Step 1 — linear transform:

    .. math::
        Z = X \cdot W + b

    Step 2 — activation (if present):

    .. math::
        A = f(Z)

    Backward pass
    -------------
    Receive upstream gradient ``dA`` (or ``dZ`` if no activation).

    Step 1 — through activation:

    .. math::
        dZ = \text{activation.backward}(dA)

    Step 2 — parameter gradients:

    .. math::
        \frac{\partial L}{\partial W} = X^T \cdot dZ
        \qquad\text{shape: } (n_{in}, n_{out})

    .. math::
        \frac{\partial L}{\partial b} = \mathbf{1}^T \cdot dZ
        = \text{sum over batch axis}
        \qquad\text{shape: } (1, n_{out})

    Step 3 — downstream gradient:

    .. math::
        dX = dZ \cdot W^T
        \qquad\text{shape: } (batch, n_{in})

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_out : int
        Number of output neurons.
    activation : Activation | None
        Activation function applied after the linear transform.
    weight_init : callable
        Initialization function for W  (default: He normal).
    seed : int | None
        Random seed for reproducibility.
    """

    trainable: bool = True

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Activation | None = None,
        weight_init: Callable[..., NDArray] = he_init,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        # ── reproducible RNG ──
        self._rng = np.random.default_rng(seed)

        # ── initialize parameters ──
        # W: (n_in, n_out)   b: (1, n_out)
        self.W: NDArray = weight_init(n_in, n_out, self._rng)
        self.b: NDArray = np.zeros((1, n_out), dtype=np.float64)

        # ── gradient buffers (same shapes) ──
        self.dW: NDArray = np.zeros_like(self.W)
        self.db: NDArray = np.zeros_like(self.b)

        # ── forward cache ──
        self._cache: dict[str, NDArray] = {}

    # ── forward ──────────────────────────────────────────────────
    def forward(self, X: NDArray) -> NDArray:
        """Compute Y = f(X · W + b).

        Parameters
        ----------
        X : ndarray, shape (batch_size, n_in)

        Returns
        -------
        Y : ndarray, shape (batch_size, n_out)
        """
        # Cache input for backward
        self._cache["X"] = X  # (batch, n_in)

        # Step 1: linear transform  Z = X @ W + b
        Z: NDArray = X @ self.W + self.b  # (batch, n_out)

        # Step 2: activation (optional)
        if self.activation is not None:
            Y: NDArray = self.activation.forward(Z)  # (batch, n_out)
        else:
            Y = Z

        return Y

    # ── backward ─────────────────────────────────────────────────
    def backward(self, dY: NDArray) -> NDArray:
        """Compute parameter gradients and downstream gradient.

        Parameters
        ----------
        dY : ndarray, shape (batch_size, n_out)
            Upstream gradient ∂L/∂Y.

        Returns
        -------
        dX : ndarray, shape (batch_size, n_in)
            Downstream gradient ∂L/∂X to propagate further.
        """
        X = self._cache["X"]  # (batch, n_in)
        batch_size = X.shape[0]

        # Step 1: back-propagate through activation
        if self.activation is not None:
            dZ: NDArray = self.activation.backward(dY)  # (batch, n_out)
        else:
            dZ = dY

        # Step 2: parameter gradients (averaged over batch)
        #   dW = (1/m) X^T @ dZ    →  (n_in, n_out)
        #   db = (1/m) Σ dZ        →  (1, n_out)
        self.dW = (X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # Step 3: downstream gradient  dX = dZ @ W^T  →  (batch, n_in)
        dX: NDArray = dZ @ self.W.T

        return dX

    # ── properties ───────────────────────────────────────────────
    @property
    def params(self) -> dict[str, NDArray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict[str, NDArray]:
        return {"W": self.dW, "b": self.db}

    def __repr__(self) -> str:
        act = self.activation.__class__.__name__ if self.activation else "None"
        return f"DenseLayer({self.n_in}, {self.n_out}, activation={act})"
