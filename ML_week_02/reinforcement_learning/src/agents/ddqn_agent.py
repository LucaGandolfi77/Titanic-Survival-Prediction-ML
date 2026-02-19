"""Double DQN agent (van Hasselt et al., 2016).

Key insight: decouple action *selection* (online network) from action
*evaluation* (target network) to reduce overestimation bias.

    target = r + γ · Q_target(s', argmax_a' Q_online(s', a'))
"""

from __future__ import annotations

from typing import Any

import torch

from .dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):
    """Double DQN — overrides only the loss computation."""

    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            # Select best actions using the *online* network …
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # … but evaluate them with the *target* network
            next_q = self.target_net(next_states).gather(1, best_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        td_errors = torch.abs(q_values - target).detach()
        loss = self.loss_fn(q_values, target)
        return loss, td_errors
