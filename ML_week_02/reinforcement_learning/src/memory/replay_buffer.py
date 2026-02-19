"""Standard uniform experience-replay buffer.

Stores transitions (s, a, r, s', done) in a fixed-size circular buffer and
samples uniformly at random for off-policy learning.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """Single environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size ring buffer with uniform sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._capacity = capacity
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Sample a random mini-batch and return tensors.

        Returns
        -------
        dict with keys:
            states      (B, state_dim)
            actions     (B, 1)       int64
            rewards     (B, 1)       float32
            next_states (B, state_dim)
            dones       (B, 1)       float32  (0/1)
        """
        transitions = random.sample(self._buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32, device=device
        )
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity
