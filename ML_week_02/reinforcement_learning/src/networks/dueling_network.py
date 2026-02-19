"""Dueling DQN architecture (Wang et al., 2016).

Decomposes Q(s, a) = V(s) + A(s, a) − mean_a A(s, a).
This allows the network to learn which states are valuable independently
of the effect of each action, which accelerates learning on environments
where many actions have similar effects in a given state.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """Dueling Q-network.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector.
    action_dim : int
        Number of discrete actions.
    hidden_dims : Sequence[int]
        Widths of the shared feature layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()

        # Shared feature extractor
        feature_layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(in_dim, h_dim))
            feature_layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        self.feature = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()

        # If only one hidden dim, use it for the streams directly
        stream_in = hidden_dims[-1] if len(hidden_dims) > 1 else state_dim
        stream_hidden = hidden_dims[-1]

        # Value stream  V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(stream_in if len(hidden_dims) > 1 else state_dim, stream_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(stream_hidden, 1),
        )

        # Advantage stream  A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(stream_in if len(hidden_dims) > 1 else state_dim, stream_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(stream_hidden, action_dim),
        )

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
        """Return Q(s, a) via the dueling decomposition.

        Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

        Parameters
        ----------
        state : Tensor  shape ``(B, state_dim)``

        Returns
        -------
        Tensor  shape ``(B, action_dim)``
        """
        features = self.feature(state)
        value = self.value_stream(features)           # (B, 1)
        advantage = self.advantage_stream(features)   # (B, action_dim)

        # Centering trick (subtract mean advantage)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
