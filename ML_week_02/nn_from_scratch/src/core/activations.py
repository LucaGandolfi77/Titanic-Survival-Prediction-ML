"""
Activation Functions — Forward & Backward Passes
=================================================

Every activation is implemented as a class with:
  • forward(Z)  → A           (element-wise or row-wise transform)
  • backward(dA) → dZ         (local gradient via chain rule)

Mathematical conventions
------------------------
  Z : pre-activation   (shape: batch × neurons)
  A : post-activation  (shape: batch × neurons)
  dA: upstream gradient ∂L/∂A
  dZ: downstream gradient ∂L/∂Z = dA ⊙ f'(Z)   (element-wise Hadamard ⊙)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Activation:
    """Abstract activation — stores cache for backward pass."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray] = {}

    def forward(self, Z: NDArray) -> NDArray:  # noqa: N803
        raise NotImplementedError

    def backward(self, dA: NDArray) -> NDArray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------
class ReLU(Activation):
    r"""Rectified Linear Unit.

    Forward
    -------
    .. math::
        A = \max(0, Z)

    Backward
    --------
    .. math::
        \frac{\partial A}{\partial Z} =
            \begin{cases}
                1 & \text{if } Z > 0 \\
                0 & \text{otherwise}
            \end{cases}

    Chain rule:
        dZ = dA ⊙ 𝟙(Z > 0)

    Shapes
    ------
    Z  : (batch_size, n_neurons)
    A  : (batch_size, n_neurons)
    dA : (batch_size, n_neurons)
    dZ : (batch_size, n_neurons)
    """

    def forward(self, Z: NDArray) -> NDArray:
        """Compute A = max(0, Z).

        Parameters
        ----------
        Z : ndarray, shape (batch_size, n_neurons)
            Pre-activation values.

        Returns
        -------
        A : ndarray, shape (batch_size, n_neurons)
            Post-activation values.
        """
        # Cache Z for backward pass
        self._cache["Z"] = Z

        # Element-wise maximum with 0
        A: NDArray = np.maximum(0, Z)
        return A

    def backward(self, dA: NDArray) -> NDArray:
        """Compute dZ = dA ⊙ 𝟙(Z > 0).

        Parameters
        ----------
        dA : ndarray, shape (batch_size, n_neurons)
            Upstream gradient ∂L/∂A.

        Returns
        -------
        dZ : ndarray, shape (batch_size, n_neurons)
            Downstream gradient ∂L/∂Z.
        """
        Z = self._cache["Z"]

        # Step 1: indicator mask — 1 where Z > 0, else 0
        mask: NDArray = (Z > 0).astype(Z.dtype)  # (batch, neurons)

        # Step 2: chain rule — element-wise Hadamard product
        dZ: NDArray = dA * mask  # (batch, neurons)
        return dZ


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------
class LeakyReLU(Activation):
    r"""Leaky Rectified Linear Unit.

    Forward
    -------
    .. math::
        A = \begin{cases}
                Z        & \text{if } Z > 0 \\
                \alpha Z & \text{otherwise}
            \end{cases}

    Backward
    --------
    .. math::
        \frac{\partial A}{\partial Z} =
            \begin{cases}
                1      & \text{if } Z > 0 \\
                \alpha & \text{otherwise}
            \end{cases}

    Chain rule:
        dZ = dA ⊙ f'(Z)

    Shapes
    ------
    Z, A, dA, dZ : (batch_size, n_neurons)
    """

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, Z: NDArray) -> NDArray:
        self._cache["Z"] = Z
        A: NDArray = np.where(Z > 0, Z, self.alpha * Z)
        return A

    def backward(self, dA: NDArray) -> NDArray:
        Z = self._cache["Z"]
        grad: NDArray = np.where(Z > 0, 1.0, self.alpha)
        dZ: NDArray = dA * grad
        return dZ

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------
class Sigmoid(Activation):
    r"""Logistic sigmoid function.

    Forward
    -------
    .. math::
        A = \sigma(Z) = \frac{1}{1 + e^{-Z}}

    Backward
    --------
    .. math::
        \frac{\partial A}{\partial Z} = A \cdot (1 - A)

    Derivation:
        Let s = σ(Z).
        ds/dZ = s · (1 − s)      (standard result)
        ⇒ dZ = dA ⊙ s ⊙ (1 − s)

    Shapes
    ------
    Z, A, dA, dZ : (batch_size, n_neurons)
    """

    def forward(self, Z: NDArray) -> NDArray:
        """Compute A = σ(Z) with numerical stability.

        Uses clipping to avoid overflow in exp(-Z).
        """
        # Clip to [-500, 500] to prevent overflow
        Z_safe: NDArray = np.clip(Z, -500, 500)
        A: NDArray = 1.0 / (1.0 + np.exp(-Z_safe))

        # Cache the output A (not Z) — backward only needs A
        self._cache["A"] = A
        return A

    def backward(self, dA: NDArray) -> NDArray:
        """Compute dZ = dA ⊙ A ⊙ (1 − A)."""
        A = self._cache["A"]

        # Local gradient: σ'(Z) = σ(Z) · (1 − σ(Z)) = A · (1 − A)
        dZ: NDArray = dA * A * (1.0 - A)
        return dZ


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------
class Tanh(Activation):
    r"""Hyperbolic tangent activation.

    Forward
    -------
    .. math::
        A = \tanh(Z) = \frac{e^{Z} - e^{-Z}}{e^{Z} + e^{-Z}}

    Backward
    --------
    .. math::
        \frac{\partial A}{\partial Z} = 1 - A^2

    Chain rule:
        dZ = dA ⊙ (1 − A²)

    Shapes
    ------
    Z, A, dA, dZ : (batch_size, n_neurons)
    """

    def forward(self, Z: NDArray) -> NDArray:
        A: NDArray = np.tanh(Z)
        self._cache["A"] = A
        return A

    def backward(self, dA: NDArray) -> NDArray:
        A = self._cache["A"]

        # tanh'(Z) = 1 − tanh²(Z) = 1 − A²
        dZ: NDArray = dA * (1.0 - A ** 2)
        return dZ


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
class Softmax(Activation):
    r"""Softmax activation (row-wise, for multi-class classification).

    Forward
    -------
    .. math::
        A_{ij} = \frac{e^{Z_{ij} - \max_k Z_{ik}}}
                      {\sum_k e^{Z_{ik} - \max_k Z_{ik}}}

    The max subtraction is for numerical stability and does not change
    the result (shift-invariance of softmax).

    Backward
    --------
    For sample *i*, the Jacobian of softmax is:

    .. math::
        \frac{\partial A_j}{\partial Z_k}
          = A_j (\delta_{jk} - A_k)

    Full Jacobian-vector product:
        dZ_i = A_i ⊙ (dA_i − (dA_i · A_i) 𝟏)

    In practice, when softmax is paired with cross-entropy loss the
    combined gradient simplifies to  dZ = A − Y  (computed in the loss).

    Shapes
    ------
    Z  : (batch_size, n_classes)
    A  : (batch_size, n_classes)   — each row sums to 1
    dA : (batch_size, n_classes)
    dZ : (batch_size, n_classes)
    """

    def forward(self, Z: NDArray) -> NDArray:
        """Numerically stable softmax: subtract row-max before exp."""
        # Step 1: shift for numerical stability  (batch, classes)
        Z_shifted: NDArray = Z - np.max(Z, axis=1, keepdims=True)

        # Step 2: exponentiate
        exp_Z: NDArray = np.exp(Z_shifted)  # (batch, classes)

        # Step 3: normalize each row
        A: NDArray = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)  # (batch, classes)

        self._cache["A"] = A
        return A

    def backward(self, dA: NDArray) -> NDArray:
        """Compute dZ via the Jacobian-vector product.

        dZ_i = A_i ⊙ (dA_i − (dA_i · A_i))

        NOTE: if paired with CrossEntropyLoss the combined backward
        returns dZ = A − Y directly, bypassing this method.
        """
        A = self._cache["A"]  # (batch, classes)

        # dot product per sample: sum(dA * A) along class axis
        dot: NDArray = np.sum(dA * A, axis=1, keepdims=True)  # (batch, 1)

        # Jacobian-vector product
        dZ: NDArray = A * (dA - dot)  # (batch, classes)
        return dZ
