#!/usr/bin/env python3
"""train_best.py — Fully train the best discovered architecture.

Usage
-----
    python scripts/train_best.py outputs/best_genome.json --epochs 100
    python scripts/train_best.py outputs/best_genome.json --epochs 200 --lr 0.001
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the best NAS architecture")
    p.add_argument("genome", type=str, help="Path to genome JSON")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="outputs/models")
    return p.parse_args()


def train_full(
    genome: Genome,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
) -> None:
    model = build_model(genome, num_classes=10).to(device)
    n_params = count_params(model)
    logger.info(f"Architecture: {genome.summary()}")
    logger.info(f"Trainable parameters: {n_params:,}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        val_split=0.1,
        num_workers=4,
        data_dir=str(_PROJ / "data"),
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  time={elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Final test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total

    logger.info(f"\nBest val accuracy:  {best_val_acc:.4f}")
    logger.info(f"Test accuracy:      {test_acc:.4f}")
    logger.info(f"Parameters:         {n_params:,}")

    # Save results
    results = {
        "genome_id": genome.id,
        "params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "history": history,
    }
    (output_dir / "training_results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    args = parse_args()
    genome = Genome.load(args.genome)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_full(genome, args.epochs, args.batch_size, args.lr, device, output_dir)


if __name__ == "__main__":
    main()
