"""
CNN Supernetwork (One-Shot Weight Sharing)
==========================================
A single large CNN whose weights are shared across all possible
sub-architectures defined by the CNN search space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupernetworkCNN(nn.Module):
    """One-shot supernet for the CNN search space.

    Allocates 5 conv blocks with max 128 filters each and a dense head
    with 512 units. Sub-architectures are evaluated by slicing channels/units.
    """

    MAX_BLOCKS = 5
    MAX_FILTERS = 128
    MAX_DENSE_LAYERS = 3
    MAX_DENSE_WIDTH = 512

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.input_conv = nn.Conv2d(in_channels, self.MAX_FILTERS, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(self.MAX_FILTERS)

        self.convs = nn.ModuleList()
        self.conv_bns = nn.ModuleList()
        for _ in range(self.MAX_BLOCKS):
            self.convs.append(nn.Conv2d(self.MAX_FILTERS, self.MAX_FILTERS, 3,
                                        padding=1, bias=False))
            self.conv_bns.append(nn.BatchNorm2d(self.MAX_FILTERS))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dense_layers = nn.ModuleList()
        self.dense_bns = nn.ModuleList()
        for _ in range(self.MAX_DENSE_LAYERS):
            self.dense_layers.append(nn.Linear(self.MAX_DENSE_WIDTH, self.MAX_DENSE_WIDTH))
            self.dense_bns.append(nn.BatchNorm1d(self.MAX_DENSE_WIDTH))

        self.input_dense = nn.Linear(self.MAX_FILTERS, self.MAX_DENSE_WIDTH)
        self.output = nn.Linear(self.MAX_DENSE_WIDTH, num_classes)

        self._active_config: Optional[Dict[str, Any]] = None

    def set_active(self, config: Dict[str, Any]) -> None:
        self._active_config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self._active_config
        if cfg is None:
            raise RuntimeError("Call set_active() before forward()")

        filters = cfg["filters"]
        n_blocks = len(filters)
        use_bn = cfg["use_batch_norm"]
        act_fn = _get_act(cfg["activation"])
        pool_fn = _get_pool(cfg["pooling_type"])
        dropout = cfg["dropout_rate"]

        # Input conv — slice to first filter count
        f0 = min(filters[0], self.MAX_FILTERS)
        w = self.input_conv.weight[:f0, :self.in_channels]
        h = F.conv2d(x, w, padding=1)
        if use_bn:
            h = F.batch_norm(h, self.input_bn.running_mean[:f0],
                             self.input_bn.running_var[:f0],
                             self.input_bn.weight[:f0], self.input_bn.bias[:f0],
                             self.training)
        h = act_fn(h)

        prev_f = f0
        for i in range(n_blocks):
            fi = min(filters[i], self.MAX_FILTERS)
            conv = self.convs[i]
            w = conv.weight[:fi, :prev_f]
            h_new = F.conv2d(h[:, :prev_f], w, padding=1)
            if use_bn:
                bn = self.conv_bns[i]
                h_new = F.batch_norm(h_new, bn.running_mean[:fi],
                                     bn.running_var[:fi],
                                     bn.weight[:fi], bn.bias[:fi], self.training)
            h_new = act_fn(h_new)
            if dropout > 0 and self.training:
                h_new = F.dropout2d(h_new, p=dropout, training=True)
            h_new = pool_fn(h_new)
            h = h_new
            prev_f = fi

        h = self.gap(h).flatten(1)

        # Project to dense width
        dw = min(cfg["dense_width"], self.MAX_DENSE_WIDTH)
        w_d = self.input_dense.weight[:dw, :prev_f]
        b_d = self.input_dense.bias[:dw]
        h = F.linear(h[:, :prev_f], w_d, b_d)
        h = act_fn(h)

        for i in range(cfg["dense_layers"]):
            layer = self.dense_layers[i]
            w = layer.weight[:dw, :dw]
            b = layer.bias[:dw]
            h = F.linear(h[:, :dw], w, b)
            if use_bn:
                bn = self.dense_bns[i]
                h = F.batch_norm(h, bn.running_mean[:dw], bn.running_var[:dw],
                                 bn.weight[:dw], bn.bias[:dw], self.training)
            h = act_fn(h)
            if dropout > 0 and self.training:
                h = F.dropout(h, p=dropout, training=True)

        w_out = self.output.weight[:, :dw]
        b_out = self.output.bias
        return F.linear(h[:, :dw], w_out, b_out)


def _get_act(name: str):
    return {"relu": torch.relu, "elu": F.elu, "gelu": F.gelu}.get(name, torch.relu)


def _get_pool(name: str):
    if name == "max":
        return lambda x: F.max_pool2d(x, 2, ceil_mode=True)
    if name == "avg":
        return lambda x: F.avg_pool2d(x, 2, ceil_mode=True)
    return lambda x: x


if __name__ == "__main__":
    supernet = SupernetworkCNN(3, 10)
    supernet.set_active({
        "filters": [16, 32], "kernel_size": 3, "use_depthwise": False,
        "use_skip_conn": False, "pooling_type": "max", "activation": "relu",
        "dropout_rate": 0.0, "use_batch_norm": False,
        "dense_layers": 1, "dense_width": 128,
    })
    x = torch.randn(4, 3, 32, 32)
    print(f"Output: {supernet(x).shape}")
