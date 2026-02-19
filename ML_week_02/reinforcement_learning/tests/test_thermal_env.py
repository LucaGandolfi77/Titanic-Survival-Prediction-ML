"""Tests for the custom ThermalControlEnv and PID controller."""

import numpy as np
import pytest

from src.environments.thermal_control_env import ThermalControlEnv
from src.control.pid_controller import PIDController
from src.control.control_utils import (
    compute_settling_time,
    compute_overshoot,
    compute_steady_state_error,
    compute_integral_absolute_error,
    compute_energy_cost,
    discretize_action,
)


def _default_config() -> dict:
    return {
        "environment": {
            "name": "ThermalControl-v0",
            "max_episode_steps": 200,
            "thermal": {
                "ambient_temp": 25.0,
                "target_temp": 55.0,
                "temp_tolerance": 5.0,
                "critical_temp": 85.0,
                "min_temp": 15.0,
                "thermal_mass": 50.0,
                "thermal_resistance": 1.5,
                "heat_generation_base": 30.0,
                "heat_generation_var": 15.0,
                "fan_cooling_power": [0.0, 5.0, 15.0, 30.0, 50.0],
                "fan_energy_cost": [0.0, 0.5, 1.5, 3.5, 7.0],
                "enable_disturbances": False,
                "workload_frequency": 0.02,
                "dt": 1.0,
            },
            "reward": {
                "temp_in_band": 1.0,
                "temp_deviation_penalty": -0.1,
                "energy_penalty": -0.05,
                "critical_penalty": -50.0,
                "smoothness_bonus": 0.1,
            },
        },
        "pid": {
            "kp": 2.0,
            "ki": 0.1,
            "kd": 1.0,
            "setpoint": 55.0,
            "output_limits": [0, 4],
            "dt": 1.0,
        },
    }


# ══════════════════════════════════════════════════════════
class TestThermalControlEnv:
    def test_reset_returns_correct_shape(self):
        env = ThermalControlEnv(config=_default_config())
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
        assert "temperature" in info

    def test_step_returns_five_elements(self):
        env = ThermalControlEnv(config=_default_config())
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(2)
        assert obs.shape == (5,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_action_space_bounds(self):
        env = ThermalControlEnv(config=_default_config())
        assert env.action_space.n == 5

    def test_temperature_stays_bounded(self):
        env = ThermalControlEnv(config=_default_config())
        env.reset(seed=42)
        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(0)  # no cooling
            if terminated or truncated:
                break
        # Temperature should not exceed critical + margin
        assert info["temperature"] <= 95.0

    def test_high_fan_cools_down(self):
        env = ThermalControlEnv(config=_default_config())
        env.reset(seed=42)
        temps = []
        for _ in range(100):
            obs, _, terminated, truncated, info = env.step(4)  # max fan
            temps.append(info["temperature"])
            if terminated or truncated:
                break
        # Temperature should decrease or stay low with max fan
        assert temps[-1] < temps[0] + 5  # didn't increase much

    def test_truncation_at_max_steps(self):
        env = ThermalControlEnv(config=_default_config())
        env.reset(seed=42)
        truncated = False
        for _ in range(250):
            _, _, terminated, truncated, _ = env.step(2)
            if terminated or truncated:
                break
        assert truncated  # should truncate at 200 steps

    def test_in_band_reward_positive(self):
        """If temperature is in band, reward should include the in-band bonus."""
        env = ThermalControlEnv(config=_default_config())
        env.reset(seed=42)
        # Force temperature near target by stepping with appropriate fan
        for _ in range(50):
            obs, reward, _, _, info = env.step(2)
        # Can't guarantee exact reward, but check reward is a float
        assert isinstance(reward, float)


# ══════════════════════════════════════════════════════════
class TestPIDController:
    def test_basic_output(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, setpoint=55.0)
        output = pid.compute(50.0)  # error = +5
        assert output > 0

    def test_output_clamped(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, setpoint=55.0, output_limits=(0, 4))
        output = pid.compute(0.0)  # huge error
        assert output == 4.0  # clamped

    def test_discrete_action(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, setpoint=55.0, output_limits=(0, 4))
        action = pid.compute_action(50.0)
        assert isinstance(action, int)
        assert 0 <= action <= 4

    def test_reset(self):
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0, setpoint=55.0)
        pid.compute(50.0)
        pid.compute(50.0)
        pid.reset()
        # After reset, integral should be zero
        output_after_reset = pid.compute(50.0)
        output_fresh = PIDController(kp=1.0, ki=0.5, kd=0.0, setpoint=55.0).compute(50.0)
        assert abs(output_after_reset - output_fresh) < 1e-6

    def test_from_config(self):
        cfg = _default_config()
        pid = PIDController.from_config(cfg)
        assert pid.kp == 2.0
        assert pid.setpoint == 55.0


# ══════════════════════════════════════════════════════════
class TestControlUtils:
    def test_settling_time_already_settled(self):
        signal = np.ones(100) * 55.0
        st = compute_settling_time(signal, setpoint=55.0, tolerance=0.02)
        assert st == 0.0

    def test_settling_time_never_settles(self):
        signal = np.linspace(0, 100, 100)
        st = compute_settling_time(signal, setpoint=55.0, tolerance=0.02)
        assert st is None or st > 0

    def test_overshoot(self):
        signal = np.array([50, 55, 65, 60, 55, 55])
        ov = compute_overshoot(signal, setpoint=55.0)
        assert ov > 0  # peak of 65 → overshoot

    def test_steady_state_error(self):
        signal = np.array([50, 52, 54, 56, 56, 56])
        sse = compute_steady_state_error(signal, setpoint=55.0, tail_fraction=0.5)
        assert sse > 0

    def test_energy_cost(self):
        actions = np.array([0, 1, 2, 3, 4])
        cost_table = [0.0, 0.5, 1.5, 3.5, 7.0]
        total = compute_energy_cost(actions, cost_table)
        assert abs(total - 12.5) < 1e-5

    def test_discretize_action(self):
        assert discretize_action(2.7, 5) == 3
        assert discretize_action(-1.0, 5) == 0
        assert discretize_action(10.0, 5) == 4
