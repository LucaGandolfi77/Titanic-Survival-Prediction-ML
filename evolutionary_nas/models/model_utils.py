"""
Model Utilities
===============
Parameter counting, size estimation, inference timing, FLOPs estimate.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    """Approximate model size in kilobytes (float32)."""
    return count_parameters(model) * 4 / 1024


def inference_time_ms(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    n_runs: int = 50,
) -> float:
    """Average inference time in milliseconds over n_runs forward passes."""
    model = model.to(device).eval()
    x = torch.randn(1, *input_shape, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000


def flops_estimate(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Rough FLOPs estimate based on multiply-accumulate operations.

    For a proper count use a profiling library; this is a fast heuristic
    that sums 2*in*out for Linear layers and K*K*Cin*Cout*H*W for Conv2d.
    """
    total = 0
    x = torch.randn(1, *input_shape)
    hooks = []

    def _hook_linear(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        nonlocal total
        total += 2 * module.in_features * module.out_features

    def _hook_conv2d(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        nonlocal total
        _, _, h, w = out.shape
        k = module.kernel_size[0] * module.kernel_size[1]
        groups = module.groups
        total += 2 * k * (module.in_channels // groups) * module.out_channels * h * w

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_hook_linear))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook_conv2d))

    with torch.no_grad():
        model.eval()
        model(x)

    for h in hooks:
        h.remove()
    return total


if __name__ == "__main__":
    from models.mlp_builder import build_mlp
    cfg = {"hidden_sizes": [128, 64], "activation": "relu",
           "dropout_rate": 0.0, "use_batch_norm": False}
    m = build_mlp(cfg, 784, 10)
    print(f"Params: {count_parameters(m):,}")
    print(f"Size: {model_size_kb(m):.1f} KB")
    print(f"FLOPs: {flops_estimate(m, (784,)):,}")
