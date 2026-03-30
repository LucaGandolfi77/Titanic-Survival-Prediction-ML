"""
Datasets
========
Load MNIST, FashionMNIST, CIFAR-10, CIFAR-100 with standard splits
and normalization. For MLP experiments, images are flattened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

DATA_ROOT = Path("data")

# Standard normalization per dataset
_NORMS = {
    "MNIST":        ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "CIFAR10":      ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR100":     ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

_DATASET_CLS = {
    "MNIST":        torchvision.datasets.MNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "CIFAR10":      torchvision.datasets.CIFAR10,
    "CIFAR100":     torchvision.datasets.CIFAR100,
}

_NUM_CLASSES = {
    "MNIST": 10, "FashionMNIST": 10, "CIFAR10": 10, "CIFAR100": 100,
}

_IN_CHANNELS = {
    "MNIST": 1, "FashionMNIST": 1, "CIFAR10": 3, "CIFAR100": 3,
}

_INPUT_DIM_FLAT = {
    "MNIST": 784, "FashionMNIST": 784, "CIFAR10": 3072, "CIFAR100": 3072,
}

_IMAGE_SIZE = {
    "MNIST": 28, "FashionMNIST": 28, "CIFAR10": 32, "CIFAR100": 32,
}


def get_dataset_info(name: str) -> dict:
    """Return num_classes, in_channels, input_dim_flat, image_size."""
    return {
        "num_classes": _NUM_CLASSES[name],
        "in_channels": _IN_CHANNELS[name],
        "input_dim_flat": _INPUT_DIM_FLAT[name],
        "image_size": _IMAGE_SIZE[name],
    }


def get_transforms(name: str, flatten: bool = False) -> Tuple:
    """Return (train_transform, test_transform)."""
    mean, std = _NORMS[name]
    base = [T.ToTensor(), T.Normalize(mean, std)]
    if flatten:
        base.append(T.Lambda(lambda x: x.view(-1)))
    train_tf = T.Compose(base)
    test_tf = T.Compose(base)
    return train_tf, test_tf


def load_dataset(
    name: str,
    batch_size: int = 64,
    flatten: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load a dataset and return (train_loader, test_loader)."""
    train_tf, test_tf = get_transforms(name, flatten=flatten)
    cls = _DATASET_CLS[name]
    root = str(DATA_ROOT / name)

    train_ds = cls(root=root, train=True, download=True, transform=train_tf)
    test_ds = cls(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    info = get_dataset_info("MNIST")
    print(f"MNIST: {info}")
    train_loader, test_loader = load_dataset("MNIST", batch_size=32)
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, labels: {y.shape}")
