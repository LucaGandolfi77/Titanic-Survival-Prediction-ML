"""Classical PID controller baseline.

Implements a discrete-time PID controller that can be used as a baseline
for comparison against the DQN agent on the thermal control environment.
"""

from __future__ import annotations

import numpy as np


class PIDController:
    """Discrete-time PID controller with output clamping.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral, derivative gains.
    setpoint : float
        Target value (e.g. desired temperature).
    output_limits : tuple[float, float]
        (min, max) bounds for the controller output.
    dt : float
        Time step (seconds).
    """

    def __init__(
        self,
        kp: float = 2.0,
        ki: float = 0.1,
        kd: float = 1.0,
        setpoint: float = 55.0,
        output_limits: tuple[float, float] = (0.0, 4.0),
        dt: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min, self.output_max = output_limits
        self.dt = dt

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset integral and derivative memory."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    # ------------------------------------------------------------------
    def compute(self, measurement: float) -> float:
        """Compute control output for the current measurement.

        Parameters
        ----------
        measurement : float
            Current process variable (e.g. temperature).

        Returns
        -------
        float
            Clamped control output.
        """
        error = self.setpoint - measurement

        # Integral (trapezoidal rule + anti-windup via clamping)
        self._integral += error * self.dt
        self._integral = np.clip(
            self._integral,
            self.output_min / max(abs(self.ki), 1e-8),
            self.output_max / max(abs(self.ki), 1e-8),
        )

        # Derivative (backward difference)
        if not self._initialized:
            derivative = 0.0
            self._initialized = True
        else:
            derivative = (error - self._prev_error) / self.dt

        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))

    # ------------------------------------------------------------------
    def compute_action(self, measurement: float) -> int:
        """Compute control output and discretise to the nearest integer action.

        Useful for environments with ``Discrete`` action spaces.
        """
        raw = self.compute(measurement)
        return int(np.clip(round(raw), int(self.output_min), int(self.output_max)))

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "PIDController":
        """Construct from a YAML config dict."""
        pid_cfg = config.get("pid", {})
        return cls(
            kp=pid_cfg.get("kp", 2.0),
            ki=pid_cfg.get("ki", 0.1),
            kd=pid_cfg.get("kd", 1.0),
            setpoint=pid_cfg.get("setpoint", 55.0),
            output_limits=tuple(pid_cfg.get("output_limits", [0, 4])),
            dt=pid_cfg.get("dt", 1.0),
        )
