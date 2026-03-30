"""
Metrics
=======
Compute accuracy, F1, latency, memory footprint, and parameter count.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from models.model_utils import count_parameters, model_size_kb, inference_time_ms


def compute_accuracy(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> float:
    """Top-1 accuracy on the provided data loader."""
    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def compute_f1(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> float:
    """Macro F1 score on the provided data loader."""
    model.to(device).eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    return float(f1_score(all_targets, all_preds, average="macro", zero_division=0))


def compute_all_metrics(
    model: nn.Module,
    loader: DataLoader,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> dict:
    """Compute a full metrics dictionary."""
    acc = compute_accuracy(model, loader, device)
    f1 = compute_f1(model, loader, device)
    params = count_parameters(model)
    size = model_size_kb(model)
    latency = inference_time_ms(model, input_shape, device="cpu")
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "param_count": params,
        "size_kb": size,
        "latency_ms": latency,
    }


if __name__ == "__main__":
    print("Metrics module ready.")
