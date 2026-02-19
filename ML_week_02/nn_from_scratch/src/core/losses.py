"""
Loss Functions — Forward (loss value) & Backward (gradient)
============================================================

Each loss class implements:
  • forward(predictions, targets) → scalar loss
  • backward()                    → gradient ∂L/∂predictions

Notation
--------
  Ŷ (Y_hat) : model predictions  — shape (batch, n_classes) or (batch, 1)
  Y         : ground-truth labels — same shape as Ŷ (one-hot or scalar)
  L         : scalar loss (averaged over batch)
  ε         : small constant for numerical stability (1e-15)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────
class Loss:
    """Abstract loss function."""

    def __init__(self) -> None:
        self._cache: dict[str, NDArray] = {}

    def forward(self, Y_hat: NDArray, Y: NDArray) -> float:
        raise NotImplementedError

    def backward(self) -> NDArray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss (multi-class, used with softmax output)
# ────────────────────────────────────────────────────────────────────
class CrossEntropyLoss(Loss):
    r"""Categorical cross-entropy loss.

    Forward
    -------
    .. math::
        L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}
            Y_{ik} \ln(\hat{Y}_{ik} + \varepsilon)

    where *m* is the batch size and *K* the number of classes.

    Backward (combined softmax + cross-entropy shortcut)
    ----------------------------------------------------
    When the preceding layer uses **softmax** activation, the combined
    gradient simplifies elegantly:

    .. math::
        \frac{\partial L}{\partial Z} = \hat{Y} - Y

    This avoids computing the full softmax Jacobian, and is both faster
    and more numerically stable.

    Shapes
    ------
    Y_hat : (batch_size, n_classes) — softmax probabilities (rows sum to 1)
    Y     : (batch_size, n_classes) — one-hot encoded targets
    dZ    : (batch_size, n_classes)
    """

    def forward(self, Y_hat: NDArray, Y: NDArray) -> float:
        """Compute mean cross-entropy loss.

        Parameters
        ----------
        Y_hat : ndarray, shape (batch, n_classes) — predicted probabilities.
        Y     : ndarray, shape (batch, n_classes) — one-hot targets.

        Returns
        -------
        loss : float — averaged scalar loss.
        """
        self._cache["Y_hat"] = Y_hat
        self._cache["Y"] = Y

        m = Y.shape[0]
        eps = 1e-15  # prevent log(0)

        # Clip predictions for numerical stability
        Y_hat_safe: NDArray = np.clip(Y_hat, eps, 1.0 - eps)

        # Cross-entropy: − (1/m) Σ Σ Y·log(Ŷ)
        loss: float = float(-np.sum(Y * np.log(Y_hat_safe)) / m)
        return loss

    def backward(self) -> NDArray:
        """Compute dZ = Ŷ − Y  (softmax + CE combined gradient).

        Returns
        -------
        dZ : ndarray, shape (batch, n_classes)
        """
        Y_hat = self._cache["Y_hat"]
        Y = self._cache["Y"]

        # Combined softmax-cross-entropy gradient
        dZ: NDArray = Y_hat - Y  # (batch, n_classes)
        return dZ


# ────────────────────────────────────────────────────────────────────
# Mean Squared Error Loss (regression)
# ────────────────────────────────────────────────────────────────────
class MSELoss(Loss):
    r"""Mean Squared Error loss for regression tasks.

    Forward
    -------
    .. math::
        L = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_i - Y_i)^2

    Backward
    --------
    .. math::
        \frac{\partial L}{\partial \hat{Y}} = \frac{2}{m} (\hat{Y} - Y)

    Shapes
    ------
    Y_hat : (batch_size, n_outputs)
    Y     : (batch_size, n_outputs)
    dY    : (batch_size, n_outputs)
    """

    def forward(self, Y_hat: NDArray, Y: NDArray) -> float:
        self._cache["Y_hat"] = Y_hat
        self._cache["Y"] = Y

        m = Y.shape[0]
        loss: float = float(np.sum((Y_hat - Y) ** 2) / m)
        return loss

    def backward(self) -> NDArray:
        Y_hat = self._cache["Y_hat"]
        Y = self._cache["Y"]
        m = Y.shape[0]

        # ∂L/∂Ŷ = 2/m · (Ŷ − Y)
        dY: NDArray = 2.0 * (Y_hat - Y) / m
        return dY


# ────────────────────────────────────────────────────────────────────
# Binary Cross-Entropy Loss (binary classification, sigmoid output)
# ────────────────────────────────────────────────────────────────────
class BinaryCrossEntropyLoss(Loss):
    r"""Binary cross-entropy loss for sigmoid outputs.

    Forward
    -------
    .. math::
        L = -\frac{1}{m} \sum_{i=1}^{m}
            \bigl[ Y_i \ln(\hat{Y}_i) + (1-Y_i) \ln(1-\hat{Y}_i) \bigr]

    Backward
    --------
    .. math::
        \frac{\partial L}{\partial \hat{Y}}
          = -\frac{1}{m} \left( \frac{Y}{\hat{Y}} - \frac{1-Y}{1-\hat{Y}} \right)

    Shapes
    ------
    Y_hat : (batch_size, 1)
    Y     : (batch_size, 1)
    dY    : (batch_size, 1)
    """

    def forward(self, Y_hat: NDArray, Y: NDArray) -> float:
        self._cache["Y_hat"] = Y_hat
        self._cache["Y"] = Y

        m = Y.shape[0]
        eps = 1e-15
        Y_hat_safe: NDArray = np.clip(Y_hat, eps, 1.0 - eps)

        loss: float = float(
            -np.sum(Y * np.log(Y_hat_safe) + (1 - Y) * np.log(1 - Y_hat_safe)) / m
        )
        return loss

    def backward(self) -> NDArray:
        Y_hat = self._cache["Y_hat"]
        Y = self._cache["Y"]
        m = Y.shape[0]
        eps = 1e-15
        Y_hat_safe: NDArray = np.clip(Y_hat, eps, 1.0 - eps)

        dY: NDArray = -(Y / Y_hat_safe - (1 - Y) / (1 - Y_hat_safe)) / m
        return dY
