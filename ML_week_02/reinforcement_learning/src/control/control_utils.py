"""Control-theory utility functions.

Provides analysis metrics borrowed from classical control:
settling time, overshoot, steady-state error, etc.
"""

from __future__ import annotations

import numpy as np


def discretize_action(continuous: float, n_levels: int) -> int:
    """Map a continuous value in [0, n_levels-1] to an integer action."""
    return int(np.clip(round(continuous), 0, n_levels - 1))


def compute_settling_time(
    signal: np.ndarray,
    setpoint: float,
    tolerance: float = 0.02,
    dt: float = 1.0,
) -> float | None:
    """Time after which *signal* stays within ±tolerance% of *setpoint*.

    Returns
    -------
    float or None
        Settling time in seconds, or ``None`` if never settles.
    """
    band = tolerance * abs(setpoint) if setpoint != 0 else tolerance
    within = np.abs(signal - setpoint) <= band

    # Walk backwards to find first exit
    for idx in range(len(within) - 1, -1, -1):
        if not within[idx]:
            if idx == len(within) - 1:
                return None  # never settled
            return float((idx + 1) * dt)
    return 0.0  # always within band


def compute_overshoot(signal: np.ndarray, setpoint: float) -> float:
    """Percentage overshoot: (peak − setpoint) / setpoint × 100."""
    peak = float(np.max(signal))
    if setpoint == 0:
        return 0.0
    overshoot = (peak - setpoint) / abs(setpoint) * 100.0
    return max(0.0, overshoot)


def compute_steady_state_error(
    signal: np.ndarray, setpoint: float, tail_fraction: float = 0.1
) -> float:
    """Average absolute error over the last *tail_fraction* of the signal."""
    n_tail = max(1, int(len(signal) * tail_fraction))
    return float(np.mean(np.abs(signal[-n_tail:] - setpoint)))


def compute_integral_absolute_error(
    signal: np.ndarray, setpoint: float, dt: float = 1.0
) -> float:
    """IAE = ∫|e(t)| dt  (trapezoidal approximation)."""
    errors = np.abs(signal - setpoint)
    return float(np.trapz(errors, dx=dt))


def compute_energy_cost(actions: np.ndarray, cost_table: list[float]) -> float:
    """Total normalised energy cost from a sequence of actions."""
    costs = np.array([cost_table[int(a)] for a in actions])
    return float(np.sum(costs))
