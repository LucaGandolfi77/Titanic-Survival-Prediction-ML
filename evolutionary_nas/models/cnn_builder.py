"""
CNN Builder
============
Construct a PyTorch nn.Module CNN from a decoded genome dictionary.
Supports depthwise-separable convolutions and skip connections.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACT_MAP = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """Single convolution block with optional BN, activation, pooling, skip."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        use_depthwise: bool,
        use_batch_norm: bool,
        activation: str,
        pooling_type: str,
        dropout_rate: float,
        use_skip: bool,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.use_skip = use_skip and (in_ch == out_ch)

        if use_depthwise and in_ch > 1:
            self.conv = DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=not use_batch_norm)

        self.bn = nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity()
        self.act = _ACT_MAP.get(activation, nn.ReLU)()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        if pooling_type == "max":
            self.pool = nn.MaxPool2d(2, ceil_mode=True)
        elif pooling_type == "avg":
            self.pool = nn.AvgPool2d(2, ceil_mode=True)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        if self.use_skip:
            out = out + x
        out = self.act(out)
        out = self.dropout(out)
        out = self.pool(out)
        return out


class DynamicCNN(nn.Module):
    """CNN with variable conv blocks and dense head, built from genome config."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: List[int],
        kernel_size: int,
        use_depthwise: bool,
        use_skip_conn: bool,
        pooling_type: str,
        activation: str,
        dropout_rate: float,
        use_batch_norm: bool,
        dense_layers: int,
        dense_width: int,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = in_channels
        for out_ch in filters:
            blocks.append(ConvBlock(
                in_ch, out_ch, kernel_size, use_depthwise,
                use_batch_norm, activation, pooling_type,
                dropout_rate, use_skip_conn,
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)

        act_cls = _ACT_MAP.get(activation, nn.ReLU)
        head: List[nn.Module] = []
        in_f = in_ch
        for _ in range(dense_layers):
            head.append(nn.Linear(in_f, dense_width))
            if use_batch_norm:
                head.append(nn.BatchNorm1d(dense_width))
            head.append(act_cls())
            if dropout_rate > 0:
                head.append(nn.Dropout(dropout_rate))
            in_f = dense_width
        head.append(nn.Linear(in_f, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def build_cnn(config: Dict[str, Any], in_channels: int, num_classes: int) -> DynamicCNN:
    """Build a CNN nn.Module from a decoded genome config dict."""
    return DynamicCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        filters=config["filters"],
        kernel_size=config["kernel_size"],
        use_depthwise=config["use_depthwise"],
        use_skip_conn=config["use_skip_conn"],
        pooling_type=config["pooling_type"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        use_batch_norm=config["use_batch_norm"],
        dense_layers=config["dense_layers"],
        dense_width=config["dense_width"],
    )


if __name__ == "__main__":
    cfg = {
        "filters": [16, 32],
        "kernel_size": 3,
        "use_depthwise": False,
        "use_skip_conn": True,
        "pooling_type": "max",
        "activation": "relu",
        "dropout_rate": 0.1,
        "use_batch_norm": True,
        "dense_layers": 1,
        "dense_width": 128,
    }
    model = build_cnn(cfg, in_channels=3, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    print(model)
    print(f"Output shape: {model(x).shape}")
