"""Build a PyTorch ``nn.Module`` from a :class:`Genome`.

The builder performs a *dry-run* shape propagation through every layer gene
so that it can calculate the correct ``in_channels`` / ``in_features`` for
each layer at construction time.  Skip connections are implemented as
element-wise additions with ``1×1`` projection convolutions when channel
dimensions mismatch.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.genome import Genome, LayerGene


# ── activation lookup ────────────────────────────────────────────────────────

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def _act(name: str) -> nn.Module:
    return _ACTIVATIONS.get(name, nn.ReLU)(inplace=True)


# ── single-layer factories ──────────────────────────────────────────────────

def _make_conv(gene: LayerGene, in_channels: int) -> Tuple[nn.Module, int]:
    """Return ``(nn.Sequential([Conv, Act]), out_channels)``."""
    ks = gene.params.get("kernel_size", 3)
    out_c = gene.params.get("filters", 64)
    act_name = gene.params.get("activation", "relu")
    pad = ks // 2  # "same" padding
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_c, ks, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        _act(act_name),
    )
    return block, out_c


def _make_pool(gene: LayerGene, pool_cls: type) -> nn.Module:
    size = gene.params.get("size", 2)
    return pool_cls(kernel_size=size, stride=size)


def _make_batchnorm(in_channels: int) -> nn.Module:
    return nn.BatchNorm2d(in_channels)


def _make_dropout(gene: LayerGene) -> nn.Module:
    rate = gene.params.get("rate", 0.3)
    return nn.Dropout2d(rate)


def _make_dense(gene: LayerGene, in_features: int) -> Tuple[nn.Module, int]:
    units = gene.params.get("units", 128)
    act_name = gene.params.get("activation", "relu")
    block = nn.Sequential(
        nn.Linear(in_features, units),
        _act(act_name),
    )
    return block, units


# ── shape tracker ────────────────────────────────────────────────────────────

def _spatial_after_pool(h: int, w: int, size: int) -> Tuple[int, int]:
    return h // size, w // size


# ── main module ──────────────────────────────────────────────────────────────

class _SkipProjection(nn.Module):
    """1×1 conv to match channel dimensions for a skip connection."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NASModel(nn.Module):
    """Dynamically-built CNN from a :class:`Genome`.

    Architecture
    ------------
    ``InputStem → [Evolvable Layers] → GlobalAvgPool → Classifier``

    * **Input stem**: fixed ``Conv2d(3, 32, 3)`` + BN + ReLU.
    * **Evolvable layers**: built from the genome.
    * **Classifier**: ``AdaptiveAvgPool2d(1) → Flatten → Linear(C, 10)``.

    Skip connections are injected as element-wise additions between the
    source and destination layer outputs.
    """

    def __init__(self, genome: Genome, num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 32, 32)) -> None:
        super().__init__()
        self.genome = genome
        self.num_classes = num_classes

        C, H, W = input_shape

        # ---- input stem (fixed) ----
        self.stem = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        cur_c, cur_h, cur_w = 32, H, W
        in_conv_mode = True  # still processing spatial feature maps

        # ---- evolvable body ----
        self.body_layers = nn.ModuleList()
        self.body_types: List[str] = []       # track type per index
        self.body_channels: List[int] = []    # channels at each layer output
        self.body_spatial: List[Tuple[int, int]] = []  # (H, W) at each layer

        # We also need to track where we transition from conv→dense
        self._flatten_idx: Optional[int] = None  # index in body_layers where flatten occurs
        cur_features: int = 0  # only used after flatten

        for i, gene in enumerate(genome.layers):
            lt = gene.layer_type

            if in_conv_mode:
                if lt == "conv2d":
                    mod, cur_c = _make_conv(gene, cur_c)
                elif lt == "maxpool":
                    if cur_h > 2 and cur_w > 2:
                        mod = _make_pool(gene, nn.MaxPool2d)
                        cur_h, cur_w = _spatial_after_pool(cur_h, cur_w, gene.params.get("size", 2))
                    else:
                        mod = nn.Identity()  # skip pool if spatial is too small
                elif lt == "avgpool":
                    if cur_h > 2 and cur_w > 2:
                        mod = _make_pool(gene, nn.AvgPool2d)
                        cur_h, cur_w = _spatial_after_pool(cur_h, cur_w, gene.params.get("size", 2))
                    else:
                        mod = nn.Identity()
                elif lt == "batchnorm":
                    mod = _make_batchnorm(cur_c)
                elif lt == "dropout":
                    mod = _make_dropout(gene)
                elif lt == "dense":
                    # Transition to FC mode
                    in_conv_mode = False
                    self._flatten_idx = i
                    cur_features = cur_c * cur_h * cur_w
                    mod, cur_features = _make_dense(gene, cur_features)
                else:
                    mod = nn.Identity()
            else:
                # Already in dense mode
                if lt == "dense":
                    mod, cur_features = _make_dense(gene, cur_features)
                elif lt == "dropout":
                    mod = nn.Dropout(gene.params.get("rate", 0.3))
                elif lt == "batchnorm":
                    mod = nn.BatchNorm1d(cur_features)
                else:
                    mod = nn.Identity()  # skip conv/pool in dense mode

            self.body_layers.append(mod)
            self.body_types.append(lt)
            self.body_channels.append(cur_c if in_conv_mode or (not in_conv_mode and self._flatten_idx == i) else cur_c)
            self.body_spatial.append((cur_h, cur_w))

        # ---- skip-connection projections ----
        self.skip_projections = nn.ModuleDict()
        self._skip_map: Dict[int, List[int]] = {}  # dst_idx → [src_indices]
        for src, dst in genome.skip_connections:
            if src >= len(self.body_layers) or dst >= len(self.body_layers):
                continue
            # Only allow skips in conv mode (before flatten)
            if self._flatten_idx is not None and (src >= self._flatten_idx or dst >= self._flatten_idx):
                continue
            self._skip_map.setdefault(dst, []).append(src)
            key = f"{src}_{dst}"
            src_c = self.body_channels[src]
            dst_c = self.body_channels[dst]
            self.skip_projections[key] = _SkipProjection(src_c, dst_c)

        # ---- classifier head ----
        if in_conv_mode:
            # No dense layers in genome: add global avg pool → linear
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(cur_c, num_classes),
            )
        else:
            self.head = nn.Linear(cur_features, num_classes)

    # ── forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        intermediates: Dict[int, torch.Tensor] = {}
        for i, (layer, lt) in enumerate(zip(self.body_layers, self.body_types)):
            # Flatten before first dense layer
            if self._flatten_idx is not None and i == self._flatten_idx:
                x = x.flatten(1)

            x = layer(x)

            # Apply skip connections targeting this layer
            if i in self._skip_map:
                for src in self._skip_map[i]:
                    key = f"{src}_{i}"
                    skip_val = self.skip_projections[key](intermediates[src])
                    # Handle spatial mismatch via adaptive pool
                    if skip_val.dim() == 4 and x.dim() == 4:
                        if skip_val.shape[2:] != x.shape[2:]:
                            skip_val = nn.functional.adaptive_avg_pool2d(
                                skip_val, x.shape[2:]
                            )
                    x = x + skip_val

            intermediates[i] = x

        return self.head(x)


# ── public helper ────────────────────────────────────────────────────────────

def build_model(
    genome: Genome,
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (3, 32, 32),
) -> NASModel:
    """Construct a :class:`NASModel` from a genome, ready for training."""
    return NASModel(genome, num_classes=num_classes, input_shape=input_shape)


def count_params(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
