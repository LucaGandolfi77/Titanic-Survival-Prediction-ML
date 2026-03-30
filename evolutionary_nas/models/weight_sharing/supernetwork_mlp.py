"""
MLP Supernetwork (One-Shot Weight Sharing)
==========================================
A single large MLP whose weights are shared across all possible sub-architectures.
Sub-architectures are sampled by selecting a prefix of layers and slicing widths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from config import CFG


class SupernetworkMLP(nn.Module):
    """One-shot supernet for the MLP search space.

    Allocates the maximum possible architecture (6 layers × 512 units)
    and shares weights so that sub-architectures can be evaluated without
    training from scratch.
    """

    MAX_LAYERS = 6
    MAX_WIDTH = 512

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, self.MAX_WIDTH)
        self.bn_input = nn.BatchNorm1d(self.MAX_WIDTH)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.MAX_LAYERS):
            self.layers.append(nn.Linear(self.MAX_WIDTH, self.MAX_WIDTH))
            self.bns.append(nn.BatchNorm1d(self.MAX_WIDTH))

        self.output_proj = nn.Linear(self.MAX_WIDTH, num_classes)

        self._active_config: Optional[Dict[str, Any]] = None

    def set_active(self, config: Dict[str, Any]) -> None:
        """Set which sub-architecture to evaluate on the next forward pass."""
        self._active_config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        cfg = self._active_config
        if cfg is None:
            raise RuntimeError("Call set_active() before forward()")

        n_layers = cfg["n_layers"]
        widths = cfg["hidden_sizes"]
        use_bn = cfg["use_batch_norm"]
        dropout = cfg["dropout_rate"]
        act_fn = _get_act(cfg["activation"])

        # Input projection, sliced to first layer width
        w0 = widths[0] if widths else self.MAX_WIDTH
        h = self.input_proj(x)[:, :w0]
        if use_bn:
            h = self.bns[0](h[:, :w0].contiguous()) if w0 == self.MAX_WIDTH else \
                nn.functional.batch_norm(
                    h, self.bns[0].running_mean[:w0], self.bns[0].running_var[:w0],
                    self.bns[0].weight[:w0], self.bns[0].bias[:w0], self.training)
        h = act_fn(h)
        if dropout > 0 and self.training:
            h = nn.functional.dropout(h, p=dropout, training=True)

        prev_w = w0
        for i in range(min(n_layers, len(widths))):
            w = widths[i]
            layer = self.layers[i]
            weight_slice = layer.weight[:w, :prev_w]
            bias_slice = layer.bias[:w]
            h = nn.functional.linear(h[:, :prev_w], weight_slice, bias_slice)
            if use_bn:
                bn = self.bns[i]
                h = nn.functional.batch_norm(
                    h, bn.running_mean[:w], bn.running_var[:w],
                    bn.weight[:w], bn.bias[:w], self.training)
            h = act_fn(h)
            if dropout > 0 and self.training:
                h = nn.functional.dropout(h, p=dropout, training=True)
            prev_w = w

        out_weight = self.output_proj.weight[:, :prev_w]
        out_bias = self.output_proj.bias
        return nn.functional.linear(h[:, :prev_w], out_weight, out_bias)


def _get_act(name: str):
    acts = {"relu": torch.relu, "tanh": torch.tanh, "elu": nn.functional.elu,
            "selu": nn.functional.selu, "gelu": nn.functional.gelu}
    return acts.get(name, torch.relu)


if __name__ == "__main__":
    supernet = SupernetworkMLP(784, 10)
    supernet.set_active({
        "n_layers": 2, "hidden_sizes": [128, 64],
        "activation": "relu", "dropout_rate": 0.0, "use_batch_norm": False,
    })
    x = torch.randn(4, 784)
    print(f"Output: {supernet(x).shape}")
