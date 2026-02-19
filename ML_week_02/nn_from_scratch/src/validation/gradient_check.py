"""
Gradient Checking — Numerical Verification of Backpropagation
=============================================================

Gradient checking compares the **analytical** gradient (from backprop)
against a **numerical** approximation using finite differences.

Numerical gradient
------------------
.. math::
    \\frac{\\partial L}{\\partial \\theta_i}
    \\approx \\frac{L(\\theta_i + \\varepsilon) - L(\\theta_i - \\varepsilon)}
                   {2 \\varepsilon}

This is the **centered difference** formula — O(ε²) accurate.

Relative error
--------------
.. math::
    \\text{rel\\_error} =
        \\frac{\\|g_{\\text{analytic}} - g_{\\text{numeric}}\\|_2}
             {\\|g_{\\text{analytic}}\\|_2 + \\|g_{\\text{numeric}}\\|_2 + \\varepsilon}

Rules of thumb:
  • rel_error < 1e-5  — ✅ correct implementation
  • rel_error < 1e-3  — ⚠️  may have a bug
  • rel_error > 1e-3  — ❌ almost certainly buggy
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def gradient_check(
    loss_fn: Callable[[NDArray], float],
    params: NDArray,
    analytic_grad: NDArray,
    epsilon: float = 1e-7,
) -> float:
    """Check gradient of a scalar loss function w.r.t. a flat parameter array.

    Parameters
    ----------
    loss_fn       : callable — takes params (flat ndarray) → scalar loss.
    params        : ndarray, shape (D,) — current parameter values.
    analytic_grad : ndarray, shape (D,) — gradient from backprop.
    epsilon       : float — perturbation size.

    Returns
    -------
    rel_error : float — relative error between numeric and analytic grads.
    """
    numeric_grad = np.zeros_like(params)

    for i in range(params.size):
        # ── perturb +ε ──
        params_plus = params.copy()
        params_plus[i] += epsilon
        loss_plus = loss_fn(params_plus)

        # ── perturb −ε ──
        params_minus = params.copy()
        params_minus[i] -= epsilon
        loss_minus = loss_fn(params_minus)

        # ── centred difference ──
        numeric_grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

    # ── relative error ──
    diff = np.linalg.norm(analytic_grad - numeric_grad)
    norm_sum = np.linalg.norm(analytic_grad) + np.linalg.norm(numeric_grad) + 1e-15
    rel_error: float = float(diff / norm_sum)

    return rel_error


def gradient_check_layer(
    layer,
    X: NDArray,
    dY: NDArray,
    loss_fn: Callable[[NDArray], float] | None = None,
    epsilon: float = 1e-7,
    verbose: bool = True,
) -> dict[str, float]:
    """Check gradients for a single trainable layer.

    Performs the forward-backward through the layer, then numerically
    verifies each parameter (W, b) using centred differences.

    Parameters
    ----------
    layer   : DenseLayer — the layer to check.
    X       : ndarray, shape (batch, n_in) — input data.
    dY      : ndarray, shape (batch, n_out) — upstream gradient.
    loss_fn : optional — if None, uses simple sum-of-output as loss.
    epsilon : perturbation size.
    verbose : print per-parameter results.

    Returns
    -------
    errors : dict[str, float] — relative error for each parameter.
    """
    # Run forward + backward to get analytic grads
    _ = layer.forward(X)
    _ = layer.backward(dY)

    errors: dict[str, float] = {}

    for param_name in layer.params:
        param = layer.params[param_name]
        analytic = layer.grads[param_name]
        numeric = np.zeros_like(param)

        # Iterate over every element in the parameter matrix
        batch_size = X.shape[0]
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index

            # ── save original, perturb +ε ──
            original = param[idx]
            param[idx] = original + epsilon
            out_plus = layer.forward(X)
            # Proxy loss averaged over batch (matching backward's 1/m)
            loss_plus = float(np.sum(out_plus * dY)) / batch_size

            # ── perturb −ε ──
            param[idx] = original - epsilon
            out_minus = layer.forward(X)
            loss_minus = float(np.sum(out_minus * dY)) / batch_size

            # ── restore ──
            param[idx] = original

            # ── centred difference ──
            numeric[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)

            it.iternext()

        # Re-run to restore cache
        _ = layer.forward(X)
        _ = layer.backward(dY)

        # Relative error for this parameter
        diff = np.linalg.norm(analytic - numeric)
        norm_sum = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-15
        rel_err = float(diff / norm_sum)
        errors[param_name] = rel_err

        if verbose:
            status = "✅" if rel_err < 1e-5 else ("⚠️" if rel_err < 1e-3 else "❌")
            print(f"  {param_name:>5s}  rel_error = {rel_err:.2e}  {status}")

    return errors
