"""
Trainer
=======
Full training loop with early stopping, LR scheduling, and
best-model checkpointing. Used for final evaluation of top architectures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0
    elapsed_seconds: float = 0.0


def _get_optimizer(
    model: nn.Module, name: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    opts = {
        "adam": torch.optim.Adam,
        "sgd": lambda p, **kw: torch.optim.SGD(p, momentum=0.9, **kw),
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    opt_cls = opts.get(name, torch.optim.Adam)
    return opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    patience: int = 7,
    checkpoint_path: Optional[Path] = None,
) -> TrainResult:
    """Train a model with early stopping and LR scheduling.

    Returns a TrainResult with full learning curves.
    """
    model = model.to(device)
    optimizer = _get_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG.LR_SCHEDULER_FACTOR,
        patience=CFG.LR_SCHEDULER_PATIENCE,
    )
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    best_val_acc = 0.0
    no_improve_count = 0
    best_state = None
    start = time.perf_counter()

    for epoch in range(epochs):
        # — Train —
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # — Validate —
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.train_accs.append(train_acc)
        result.val_accs.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.debug(f"Early stopping at epoch {epoch+1}")
            break

    result.best_val_acc = best_val_acc
    result.best_epoch = result.val_accs.index(best_val_acc) + 1
    result.total_epochs = len(result.train_losses)
    result.elapsed_seconds = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)

    return result


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


if __name__ == "__main__":
    print("Trainer module ready.")
