"""
Weight Initializers
===================

Proper initialization is *critical* for training deep networks.
All initializers follow the pattern:

    W = init_fn(fan_in, fan_out, rng) → ndarray, shape (fan_in, fan_out)

where ``fan_in`` = number of input units and ``fan_out`` = number of output
units for a given layer.

Terminology
-----------
  fan_in  (n_in)  : dimensionality of the input
  fan_out (n_out) : dimensionality of the output
  rng             : numpy.random.Generator for reproducibility
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def xavier_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> NDArray:
    r"""Glorot / Xavier uniform initialization.

    .. math::
        W \sim \mathcal{U}\!\left[
            -\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\;
             \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
        \right]

    Motivation
    ----------
    Keeps the variance of activations and gradients roughly equal across
    layers when using *linear* or *tanh* activations.

    Reference: Glorot & Bengio, 2010.

    Parameters
    ----------
    fan_in  : int — number of input neurons.
    fan_out : int — number of output neurons.
    rng     : Generator, optional — PRNG for reproducibility.

    Returns
    -------
    W : ndarray, shape (fan_in, fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    limit: float = np.sqrt(6.0 / (fan_in + fan_out))
    W: NDArray = rng.uniform(-limit, limit, size=(fan_in, fan_out))
    return W.astype(np.float64)


def he_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> NDArray:
    r"""He (Kaiming) normal initialization.

    .. math::
        W \sim \mathcal{N}\!\left(0,\; \sqrt{\frac{2}{n_{\text{in}}}}\right)

    Motivation
    ----------
    Derived for ReLU activations: compensates for the fact that ReLU zeroes
    out half the activations, so variance should be doubled → factor of 2.

    Reference: He et al., 2015.

    Parameters
    ----------
    fan_in  : int — number of input neurons.
    fan_out : int — number of output neurons.
    rng     : Generator, optional.

    Returns
    -------
    W : ndarray, shape (fan_in, fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    std: float = np.sqrt(2.0 / fan_in)
    W: NDArray = rng.normal(0.0, std, size=(fan_in, fan_out))
    return W.astype(np.float64)


def lecun_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> NDArray:
    r"""LeCun normal initialization.

    .. math::
        W \sim \mathcal{N}\!\left(0,\; \sqrt{\frac{1}{n_{\text{in}}}}\right)

    Motivation
    ----------
    Originally proposed for *SELU* (scaled exponential linear unit) and
    sigmoid-family activations where half-zeroing does not apply.

    Reference: LeCun et al., 1998.

    Parameters
    ----------
    fan_in  : int — number of input neurons.
    fan_out : int — number of output neurons.
    rng     : Generator, optional.

    Returns
    -------
    W : ndarray, shape (fan_in, fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    std: float = np.sqrt(1.0 / fan_in)
    W: NDArray = rng.normal(0.0, std, size=(fan_in, fan_out))
    return W.astype(np.float64)


def zeros_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """All-zeros initialization (typically used for biases).

    Parameters
    ----------
    fan_in  : int
    fan_out : int
    rng     : ignored — included for API consistency.

    Returns
    -------
    W : ndarray, shape (fan_in, fan_out), all zeros.
    """
    return np.zeros((fan_in, fan_out), dtype=np.float64)
