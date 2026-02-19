"""
Optimizers — Parameter Update Rules
====================================

Each optimizer maintains per-parameter state (e.g., velocity, moments) and
updates weights in-place via ``step(params, grads)``.

Notation
--------
  θ   : parameter (W or b)
  g   : gradient ∂L/∂θ
  η   : learning rate (lr)
  v   : velocity / first-moment estimate
  s   : second-moment estimate
  β₁  : exponential decay rate for first moment  (Adam)
  β₂  : exponential decay rate for second moment (Adam)
  ε   : small constant to prevent division by zero
  t   : time-step counter (for bias correction in Adam)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────
class Optimizer:
    """Abstract optimizer."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, layers: list) -> None:  # type: ignore[type-arg]
        """Update parameters of every trainable layer."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"


# ────────────────────────────────────────────────────────────────────
# Vanilla SGD
# ────────────────────────────────────────────────────────────────────
class SGD(Optimizer):
    r"""Stochastic Gradient Descent (vanilla, no momentum).

    Update rule
    -----------
    .. math::
        \theta \leftarrow \theta - \eta \, g

    Parameters
    ----------
    lr : float — learning rate η.
    """

    def step(self, layers: list) -> None:
        """Apply vanilla SGD update to all trainable layers.

        Parameters
        ----------
        layers : list[Layer]
            Layers of the network; only those with ``trainable=True``
            are updated.
        """
        for layer in layers:
            if not layer.trainable:
                continue
            params = layer.params  # {"W": ..., "b": ...}
            grads = layer.grads    # {"W": ..., "b": ...}
            for key in params:
                # θ ← θ − η · g
                params[key] -= self.lr * grads[key]


# ────────────────────────────────────────────────────────────────────
# SGD with Momentum
# ────────────────────────────────────────────────────────────────────
class Momentum(Optimizer):
    r"""SGD with classical momentum.

    Update rule
    -----------
    .. math::
        v &\leftarrow \beta \, v + (1 - \beta) \, g \\
        \theta &\leftarrow \theta - \eta \, v

    Some implementations use ``v ← β v + g`` then ``θ ← θ − η v``.
    Both are equivalent up to a scaling of η.  We use the exponential
    moving average form for consistency with Adam.

    Parameters
    ----------
    lr   : float — learning rate η  (default: 0.01).
    beta : float — momentum coefficient β  (default: 0.9).
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self._velocity: dict[int, dict[str, NDArray]] = {}

    def step(self, layers: list) -> None:
        for layer in layers:
            if not layer.trainable:
                continue

            lid = id(layer)
            if lid not in self._velocity:
                self._velocity[lid] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }

            params = layer.params
            grads = layer.grads
            vel = self._velocity[lid]

            for key in params:
                # v ← β v + (1 − β) g
                vel[key] = self.beta * vel[key] + (1.0 - self.beta) * grads[key]
                # θ ← θ − η v
                params[key] -= self.lr * vel[key]

    def __repr__(self) -> str:
        return f"Momentum(lr={self.lr}, beta={self.beta})"


# ────────────────────────────────────────────────────────────────────
# Adam
# ────────────────────────────────────────────────────────────────────
class Adam(Optimizer):
    r"""Adam optimizer (Adaptive Moment Estimation).

    Update rule
    -----------
    .. math::
        v  &\leftarrow \beta_1 \, v + (1 - \beta_1) \, g
           \qquad\text{(first moment — mean of gradients)} \\
        s  &\leftarrow \beta_2 \, s + (1 - \beta_2) \, g^2
           \qquad\text{(second moment — uncentered variance)} \\
        \hat{v} &= \frac{v}{1 - \beta_1^t}
           \qquad\text{(bias-corrected first moment)} \\
        \hat{s} &= \frac{s}{1 - \beta_2^t}
           \qquad\text{(bias-corrected second moment)} \\
        \theta &\leftarrow \theta
          - \eta \frac{\hat{v}}{\sqrt{\hat{s}} + \varepsilon}

    Parameters
    ----------
    lr    : float — learning rate η   (default: 0.001).
    beta1 : float — β₁ for first moment  (default: 0.9).
    beta2 : float — β₂ for second moment (default: 0.999).
    eps   : float — ε for numerical stability (default: 1e-8).

    Reference: Kingma & Ba, 2015.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t: int = 0                                         # global time-step
        self._v: dict[int, dict[str, NDArray]] = {}               # first moments
        self._s: dict[int, dict[str, NDArray]] = {}               # second moments

    def step(self, layers: list) -> None:
        self._t += 1  # increment time-step once per call

        for layer in layers:
            if not layer.trainable:
                continue

            lid = id(layer)
            if lid not in self._v:
                self._v[lid] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }
                self._s[lid] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }

            params = layer.params
            grads = layer.grads
            v = self._v[lid]
            s = self._s[lid]

            for key in params:
                g = grads[key]

                # Step 1: update biased first moment estimate
                # v ← β₁ v + (1 − β₁) g
                v[key] = self.beta1 * v[key] + (1.0 - self.beta1) * g

                # Step 2: update biased second moment estimate
                # s ← β₂ s + (1 − β₂) g²
                s[key] = self.beta2 * s[key] + (1.0 - self.beta2) * (g ** 2)

                # Step 3: bias correction
                # v̂ = v / (1 − β₁ᵗ)
                v_hat: NDArray = v[key] / (1.0 - self.beta1 ** self._t)
                # ŝ = s / (1 − β₂ᵗ)
                s_hat: NDArray = s[key] / (1.0 - self.beta2 ** self._t)

                # Step 4: parameter update
                # θ ← θ − η · v̂ / (√ŝ + ε)
                params[key] -= self.lr * v_hat / (np.sqrt(s_hat) + self.eps)

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )
