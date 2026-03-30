"""
MLP Builder
============
Construct a PyTorch nn.Module MLP from a decoded genome dictionary.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


_ACT_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}


class DynamicMLP(nn.Module):
    """MLP with variable depth, width, activation, batchnorm, and dropout."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        act_cls = _ACT_MAP.get(activation, nn.ReLU)
        layers: List[nn.Module] = []
        in_features = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def build_mlp(config: Dict[str, Any], input_dim: int, num_classes: int) -> DynamicMLP:
    """Build an MLP nn.Module from a decoded genome config dict."""
    return DynamicMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        use_batch_norm=config["use_batch_norm"],
    )


if __name__ == "__main__":
    cfg = {
        "hidden_sizes": [128, 64],
        "activation": "relu",
        "dropout_rate": 0.2,
        "use_batch_norm": True,
    }
    model = build_mlp(cfg, input_dim=784, num_classes=10)
    x = torch.randn(4, 784)
    print(model)
    print(f"Output shape: {model(x).shape}")
