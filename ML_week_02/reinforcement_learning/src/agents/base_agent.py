"""Abstract base agent for all DQN variants."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """Interface that every DQN variant must implement.

    All concrete agents share:
    * An online Q-network and a target Q-network.
    * An experience-replay buffer.
    * Epsilon-greedy exploration with configurable decay.
    * Checkpoint save / load utilities.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        # Exploration parameters
        agent_cfg = config["agent"]
        self.epsilon = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = agent_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = agent_cfg.get("epsilon_decay", 0.995)

        self.state_dim: int = agent_cfg["state_dim"]
        self.action_dim: int = agent_cfg["action_dim"]
        self.gamma: float = agent_cfg.get("gamma", 0.99)
        self.batch_size: int = agent_cfg.get("batch_size", 64)
        self.gradient_clip: float = agent_cfg.get("gradient_clip", 1.0)
        self.learning_starts: int = agent_cfg.get("learning_starts", 1000)
        self.train_frequency: int = agent_cfg.get("train_frequency", 4)
        self.target_update_frequency: int = agent_cfg.get("target_update_frequency", 10)
        self.use_soft_update: bool = agent_cfg.get("use_soft_update", False)
        self.soft_update_tau: float = agent_cfg.get("soft_update_tau", 0.005)

        # Total env-steps taken (drives learning_starts / train_frequency)
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        """Choose action for *state* (epsilon-greedy in train, greedy in eval)."""

    @abc.abstractmethod
    def train_step(self) -> dict[str, float] | None:
        """One gradient update taken from the replay buffer.

        Returns a dict of logged metrics (e.g. ``{'loss': …}``) or ``None``
        if the buffer does not yet have enough samples.
        """

    @abc.abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay, clamped to `epsilon_end`."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def hard_update_target(self) -> None:
        """Copy online-network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target(self) -> None:
        """Polyak average: θ_target ← τ·θ_online + (1−τ)·θ_target."""
        tau = self.soft_update_tau
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def update_target(self) -> None:
        if self.use_soft_update:
            self.soft_update_target()
        else:
            self.hard_update_target()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.total_steps = checkpoint.get("total_steps", 0)
        return checkpoint
