"""
Fast Trainer
============
Reduced training (few epochs) for fitness estimation in the evolutionary
inner loop. Returns partial learning curves for predictive early stopping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG


@dataclass
class FastTrainResult:
    val_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    final_val_acc: float = 0.0
    epochs_run: int = 0
    elapsed_seconds: float = 0.0
    early_stopped: bool = False


def fast_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    learning_curve_predictor: Optional[object] = None,
    accuracy_threshold: float = 0.0,
) -> FastTrainResult:
    """Train for a few epochs and return the accuracy trajectory.

    If a learning_curve_predictor is provided, training may be aborted
    early if the predicted final accuracy is below threshold.
    """
    model = model.to(device)
    optimizer = _get_optimizer(model, optimizer_name, lr, weight_decay)
    criterion = nn.CrossEntropyLoss()

    result = FastTrainResult()
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        result.val_accs.append(val_acc)
        result.val_losses.append(val_loss)

        # Predictive early stopping
        if (learning_curve_predictor is not None and
                len(result.val_accs) >= CFG.PREDICTOR_LOOKBACK):
            predicted_final = learning_curve_predictor.predict(result.val_accs)
            if predicted_final < accuracy_threshold:
                result.early_stopped = True
                break

    result.final_val_acc = result.val_accs[-1] if result.val_accs else 0.0
    result.epochs_run = len(result.val_accs)
    result.elapsed_seconds = time.perf_counter() - start
    return result


def _get_optimizer(model, name, lr, wd):
    opts = {
        "adam": torch.optim.Adam,
        "sgd": lambda p, **kw: torch.optim.SGD(p, momentum=0.9, **kw),
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    return opts.get(name, torch.optim.Adam)(
        model.parameters(), lr=lr, weight_decay=wd
    )


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            loss_sum += criterion(out, targets).item() * inputs.size(0)
            correct += (out.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


class LearningCurvePredictor:
    """Simple MLP-based predictor that estimates final accuracy from
    a partial learning curve (first K epochs).

    Uses a small feedforward net trained on historical curves.
    """

    def __init__(self, lookback: int = 3, hidden: int = 32):
        self.lookback = lookback
        self.hidden = hidden
        self.model: Optional[nn.Module] = None
        self._curves: List[List[float]] = []
        self._finals: List[float] = []

    def add_curve(self, partial: List[float], final_acc: float) -> None:
        """Record a training curve for later fitting."""
        if len(partial) >= self.lookback:
            self._curves.append(partial[:self.lookback])
            self._finals.append(final_acc)

    def fit(self) -> None:
        """Train the predictor on accumulated curves."""
        if len(self._curves) < 10:
            return
        X = torch.tensor(self._curves, dtype=torch.float32)
        y = torch.tensor(self._finals, dtype=torch.float32).unsqueeze(1)

        self.model = nn.Sequential(
            nn.Linear(self.lookback, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for _ in range(200):
            opt.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

    def predict(self, partial_curve: List[float]) -> float:
        """Predict final accuracy from a partial curve."""
        if self.model is None or len(partial_curve) < self.lookback:
            return partial_curve[-1] if partial_curve else 0.0
        x = torch.tensor(
            partial_curve[-self.lookback:], dtype=torch.float32
        ).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return self.model(x).item()

    @property
    def n_curves(self) -> int:
        return len(self._curves)


if __name__ == "__main__":
    predictor = LearningCurvePredictor(lookback=3)
    for i in range(20):
        curve = [0.1 + 0.05 * j + 0.01 * i for j in range(5)]
        predictor.add_curve(curve[:3], curve[-1])
    predictor.fit()
    test_curve = [0.15, 0.22, 0.28]
    print(f"Predicted final: {predictor.predict(test_curve):.4f}")
