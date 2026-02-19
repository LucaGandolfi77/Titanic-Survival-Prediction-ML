"""Environment wrappers (Gymnasium-style).

Provides composable wrappers for reward scaling, observation
normalisation, and episode statistics collection.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale


class EpisodeStatsWrapper(gym.Wrapper):
    """Track per-episode statistics (return, length).

    After each ``done``, the ``info`` dict contains::

        info["episode"] = {"r": total_return, "l": length}
    """

    def __init__(self, env: gym.Env, window: int = 100) -> None:
        super().__init__(env)
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self.return_history: deque[float] = deque(maxlen=window)
        self.length_history: deque[int] = deque(maxlen=window)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        self._episode_return = 0.0
        self._episode_length = 0
        return super().reset(**kwargs)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._episode_return += reward
        self._episode_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_length,
            }
            self.return_history.append(self._episode_return)
            self.length_history.append(self._episode_length)

        return obs, reward, terminated, truncated, info


class RunningNormWrapper(gym.ObservationWrapper):
    """Normalise observations with a running mean/std (Welford's algorithm).

    Useful for environments where the observation scale is unknown a priori.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0) -> None:
        super().__init__(env)
        shape = env.observation_space.shape
        self._mean = np.zeros(shape, dtype=np.float64)
        self._var = np.ones(shape, dtype=np.float64)
        self._count: int = 0
        self._clip = clip

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update(obs)
        normalised = (obs - self._mean) / (np.sqrt(self._var) + 1e-8)
        return np.clip(normalised, -self._clip, self._clip).astype(np.float32)

    def _update(self, obs: np.ndarray) -> None:
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += (delta * delta2 - self._var) / self._count
