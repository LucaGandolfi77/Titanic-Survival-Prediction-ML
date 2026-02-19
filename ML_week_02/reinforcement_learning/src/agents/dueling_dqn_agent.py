"""Dueling DQN agent (Wang et al., 2016).

Combines the Double-DQN target-computation trick with the Dueling
network architecture that separates state-value and advantage streams.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..memory.replay_buffer import ReplayBuffer
from ..memory.prioritized_buffer import PrioritizedReplayBuffer
from ..networks.dueling_network import DuelingNetwork
from .base_agent import BaseAgent


class DuelingDQNAgent(BaseAgent):
    """Dueling DQN with Double-DQN targets.

    Parameters
    ----------
    config : dict
        Full experiment configuration.
    device : torch.device
        Compute device.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        agent_cfg = config["agent"]

        hidden_dims = agent_cfg.get("hidden_dims", [128, 128])

        # Networks — Dueling architecture
        self.online_net = DuelingNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        self.target_net = DuelingNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        self.hard_update_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=agent_cfg.get("learning_rate", 1e-3)
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # Replay buffer
        buffer_type = agent_cfg.get("buffer_type", "standard")
        seed = config.get("experiment", {}).get("seed")
        if buffer_type == "prioritized":
            self.memory: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                capacity=agent_cfg.get("buffer_size", 100_000),
                alpha=agent_cfg.get("per_alpha", 0.6),
                seed=seed,
            )
            self._per_beta = agent_cfg.get("per_beta_start", 0.4)
            self._per_beta_end = agent_cfg.get("per_beta_end", 1.0)
            self._per_beta_anneal = agent_cfg.get("per_beta_anneal_episodes", 400)
        else:
            self.memory = ReplayBuffer(
                capacity=agent_cfg.get("buffer_size", 100_000), seed=seed
            )
            self._per_beta = None

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    def train_step(self) -> dict[str, float] | None:
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return None

        is_per = isinstance(self.memory, PrioritizedReplayBuffer)
        if is_per:
            batch = self.memory.sample(self.batch_size, beta=self._per_beta, device=self.device)
        else:
            batch = self.memory.sample(self.batch_size, device=self.device)

        loss, td_errors = self._compute_loss(batch)

        if is_per:
            loss = (loss * batch["weights"]).mean()
        else:
            loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        if is_per:
            self.memory.update_priorities(batch["indices"], td_errors)

        return {"loss": loss.item()}

    # ------------------------------------------------------------------
    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Double-DQN style target with Dueling networks."""
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        td_errors = torch.abs(q_values - target).detach()
        loss = self.loss_fn(q_values, target)
        return loss, td_errors

    # ------------------------------------------------------------------
    def anneal_per_beta(self, episode: int) -> None:
        if self._per_beta is not None:
            frac = min(episode / max(self._per_beta_anneal, 1), 1.0)
            self._per_beta = (
                self._per_beta + frac * (self._per_beta_end - self._per_beta)
            )
