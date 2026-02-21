#!/usr/bin/env python3
"""compare_resnet.py — Compare the NAS-discovered architecture vs ResNet-18.

Usage
-----
    python scripts/compare_resnet.py outputs/best_genome.json
    python scripts/compare_resnet.py outputs/best_genome.json --epochs 50
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TMPDIR", "/tmp")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from src.builder import build_model, count_params
from src.genome import Genome
from src.trainer import get_cifar10_loaders, get_device


def _train_and_eval(
    model: nn.Module,
    name: str,
    epochs: int,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    lr: float = 0.01,
) -> Dict:
    """Train *model* and return result dict."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n{'─'*50}")
    logger.info(f"Training {name}  ({n_params:,} params)")
    logger.info(f"{'─'*50}")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_sd = None

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if val_acc > best_val:
            best_val = val_acc
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"  {name} epoch {epoch}/{epochs}  val_acc={val_acc:.4f}")

    # Test
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total

    return {
        "name": name,
        "params": n_params,
        "best_val_acc": best_val,
        "test_acc": test_acc,
    }


def _build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10 (32×32 input, no initial pooling)."""
    import torchvision.models as models

    model = models.resnet18(weights=None, num_classes=num_classes)
    # Replace first conv (7×7 stride 2 → 3×3 stride 1) for 32×32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # remove the max-pool for CIFAR
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare NAS architecture vs ResNet-18")
    p.add_argument("genome", type=str, help="Path to best genome JSON")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="outputs/comparison.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    genome = Genome.load(args.genome)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=4,
        data_dir=str(_PROJ / "data"),
    )

    # 1. NAS model
    nas_model = build_model(genome, num_classes=10).to(device)
    nas_result = _train_and_eval(
        nas_model, "NAS-Best", args.epochs,
        train_loader, val_loader, test_loader, device,
    )

    # 2. ResNet-18
    resnet = _build_resnet18_cifar().to(device)
    resnet_result = _train_and_eval(
        resnet, "ResNet-18", args.epochs,
        train_loader, val_loader, test_loader, device,
    )

    # Report
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*60}")
    for r in [nas_result, resnet_result]:
        logger.info(
            f"  {r['name']:12s} | params={r['params']:>10,} | "
            f"val={r['best_val_acc']:.4f} | test={r['test_acc']:.4f}"
        )

    diff = nas_result["test_acc"] - resnet_result["test_acc"]
    logger.info(f"\nNAS vs ResNet-18 test gap: {diff:+.4f}")
    logger.info(
        f"NAS param ratio: {nas_result['params'] / resnet_result['params']:.2f}× ResNet-18"
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"nas": nas_result, "resnet18": resnet_result}, indent=2))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
