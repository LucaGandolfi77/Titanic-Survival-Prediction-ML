"""Standard DQN Q-network (MLP).

Architecture follows the pattern from Mnih et al. (2015) adapted for
low-dimensional state inputs (e.g. CartPole).  For image-based envs a
convolutional front-end would be prepended — that extension is left as a
future exercise.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Fully-connected Q-network.

    Maps *state* → Q(s, a) for every action simultaneously.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector.
    action_dim : int
        Number of discrete actions.
    hidden_dims : Sequence[int]
        Widths of hidden layers (e.g. ``[128, 128]``).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Initialise weights (He / Kaiming)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action.

        Parameters
        ----------
        state : Tensor  shape ``(B, state_dim)``

        Returns
        -------
        Tensor  shape ``(B, action_dim)``
        """
        return self.net(state)
