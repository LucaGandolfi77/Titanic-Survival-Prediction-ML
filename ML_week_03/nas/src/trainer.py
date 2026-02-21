"""Train a single CNN architecture and return its fitness (val accuracy).

Supports:
* Configurable epochs, batch size, optimiser, scheduler.
* **Early stopping** — kill bad architectures at epoch *N* if accuracy is
  below a threshold (saves massive compute).
* Apple Silicon MPS backend, CUDA, or CPU fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from src.builder import NASModel, build_model, count_params
from src.genome import Genome


# ── device selection ─────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    """Pick the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ── data helpers ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 2,
    data_dir: str = "data",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` for CIFAR-10.

    The validation set is carved from the training split.
    """
    import torchvision
    import torchvision.transforms as T

    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_tf = T.Compose([T.ToTensor(), normalize])

    test_tf = T.Compose([T.ToTensor(), normalize])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # MPS does not support pin_memory yet — disable it to silence the warning
    _pin = not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    loader_kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


# ── training loop ────────────────────────────────────────────────────────────

class EarlyStopSignal(Exception):
    """Raised when an architecture is killed early."""


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return top-1 accuracy ∈ [0, 1]."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_genome(
    genome: Genome,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
    existing_state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train *genome* and return a result dict.

    Parameters
    ----------
    genome : Genome
        Architecture to evaluate.
    train_loader, val_loader : DataLoader
        Data loaders (may be shared across workers via ``fork``).
    cfg : dict
        The full YAML config (or at least ``training`` + ``device`` keys).
    device : torch.device, optional
        Override device selection.
    existing_state_dict : dict, optional
        Pre-trained weights to warm-start (weight inheritance).

    Returns
    -------
    dict
        ``{"fitness": float, "epochs_trained": int, "params": int,
        "history": list[dict], "early_stopped": bool}``
    """
    if device is None:
        device = get_device(cfg.get("device", "auto"))

    tcfg = cfg.get("training", {})
    epochs = tcfg.get("epochs", 10)
    lr = tcfg.get("lr", 0.01)
    momentum = tcfg.get("momentum", 0.9)
    wd = tcfg.get("weight_decay", 1e-4)

    es_cfg = tcfg.get("early_stop", {})
    es_enabled = es_cfg.get("enabled", True)
    es_epoch = es_cfg.get("patience_epoch", 3)
    es_min_acc = es_cfg.get("min_accuracy", 0.40)

    # Build model
    try:
        model = build_model(genome, num_classes=10).to(device)
    except Exception as exc:
        logger.warning(f"Genome {genome.id}: build failed — {exc}")
        return {
            "fitness": 0.0,
            "epochs_trained": 0,
            "params": 0,
            "history": [],
            "early_stopped": True,
        }

    n_params = count_params(model)

    # Optionally load inherited weights
    if existing_state_dict is not None:
        _load_partial(model, existing_state_dict)

    # Optimiser & scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    history = []
    early_stopped = False

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        if scheduler is not None:
            scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]

        # ── validate ──
        val_acc = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})

        logger.debug(
            f"  Genome {genome.id} | epoch {epoch}/{epochs} | "
            f"loss={avg_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # ── early-stop check ──
        if es_enabled and epoch == es_epoch and val_acc < es_min_acc:
            logger.info(
                f"  Genome {genome.id} killed early at epoch {epoch} "
                f"(val_acc={val_acc:.4f} < {es_min_acc})"
            )
            early_stopped = True
            break

    fitness = history[-1]["val_acc"] if history else 0.0
    genome.fitness = fitness

    return {
        "fitness": fitness,
        "epochs_trained": len(history),
        "params": n_params,
        "history": history,
        "early_stopped": early_stopped,
        "state_dict": model.state_dict(),
    }


# ── partial weight loading (for weight inheritance) ──────────────────────────

def _load_partial(model: nn.Module, state_dict: Dict[str, Any]) -> int:
    """Load matching keys from *state_dict* into *model*.

    Returns the number of tensors successfully loaded.
    """
    own = model.state_dict()
    loaded = 0
    for key, tensor in state_dict.items():
        if key in own and own[key].shape == tensor.shape:
            own[key].copy_(tensor)
            loaded += 1
    model.load_state_dict(own, strict=False)
    return loaded
