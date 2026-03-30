"""
Path Sampler
=============
Sample sub-architectures from the supernet and evaluate them using
inherited weights. This is the core of one-shot NAS.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from search_space.genome_encoder import decode
from models.weight_sharing.supernetwork_mlp import SupernetworkMLP
from models.weight_sharing.supernetwork_cnn import SupernetworkCNN


def evaluate_subnet(
    supernet: nn.Module,
    genome: list[float],
    net_type: str,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, int]:
    """Evaluate a sub-architecture by setting it active in the supernet.

    Returns:
        (accuracy, param_count_estimate)
    """
    config = decode(genome, net_type)
    supernet.set_active(config)
    supernet.to(device).eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = supernet(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / max(total, 1)
    param_count = _estimate_subnet_params(config, net_type)
    return accuracy, param_count


def _estimate_subnet_params(config: Dict[str, Any], net_type: str) -> int:
    """Estimate param count of a sub-architecture without building it."""
    total = 0
    if net_type == "mlp":
        sizes = config["hidden_sizes"]
        # Assume input_dim and num_classes are unknown; estimate relative cost
        prev = 784  # placeholder; overridden by actual input_dim in practice
        for s in sizes:
            total += prev * s + s  # weight + bias
            if config["use_batch_norm"]:
                total += 2 * s
            prev = s
        total += prev * 10 + 10
    elif net_type == "cnn":
        k = config["kernel_size"]
        prev_ch = 3
        for f in config["filters"]:
            if config.get("use_depthwise"):
                total += prev_ch * k * k + prev_ch * f
            else:
                total += prev_ch * f * k * k + f
            if config["use_batch_norm"]:
                total += 2 * f
            prev_ch = f
        prev = prev_ch
        dw = config["dense_width"]
        total += prev * dw + dw
        for _ in range(config["dense_layers"]):
            total += dw * dw + dw
            if config["use_batch_norm"]:
                total += 2 * dw
        total += dw * 10 + 10
    return total


def train_supernet(
    supernet: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
    sample_fn=None,
) -> float:
    """Train the supernet for a number of epochs, sampling random
    sub-architectures at each step.

    Args:
        sample_fn: callable that returns (genome, net_type) tuples for
                   randomly sampling architectures. If None, supernet
                   must already have set_active called.

    Returns:
        Average training loss over the last epoch.
    """
    supernet.to(device).train()
    optimizer = torch.optim.Adam(supernet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if sample_fn is not None:
                genome, net_type = sample_fn()
                config = decode(genome, net_type)
                supernet.set_active(config)
            optimizer.zero_grad()
            outputs = supernet(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        last_loss = running_loss / max(n_batches, 1)
    return last_loss


if __name__ == "__main__":
    print("Path sampler ready.")
