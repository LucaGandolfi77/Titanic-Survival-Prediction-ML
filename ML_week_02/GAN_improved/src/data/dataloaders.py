"""
Dataset wrappers and dataloader factories.

Supports: MNIST, Fashion-MNIST, CIFAR-10.
Handles automatic download, normalization to [-1, 1], and optional resizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Registry of supported datasets
_DATASET_REGISTRY: dict[str, type[datasets.VisionDataset]] = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def get_transform(
    image_size: int = 28,
    n_channels: int = 1,
) -> transforms.Compose:
    """Build the standard GAN transform pipeline.

    - Resize to target size
    - Convert to tensor [0, 1]
    - Normalize to [-1, 1] (for Tanh output)

    Args:
        image_size: Target spatial resolution.
        n_channels: Number of image channels (1=grayscale, 3=RGB).

    Returns:
        torchvision Compose transform.
    """
    transform_list: list[Any] = [transforms.Resize(image_size)]

    if n_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * n_channels,
            std=[0.5] * n_channels,
        ),
    ])

    return transforms.Compose(transform_list)


def get_dataset(
    dataset_name: str = "mnist",
    data_dir: str | Path = "data/datasets",
    image_size: int = 28,
    n_channels: int = 1,
    train: bool = True,
) -> Dataset:
    """Get a torchvision dataset with standard GAN preprocessing.

    Args:
        dataset_name: One of 'mnist', 'fashion_mnist', 'cifar10'.
        data_dir: Root directory for dataset storage.
        image_size: Target spatial resolution.
        n_channels: Number of output channels.
        train: Whether to load the training split.

    Returns:
        A torchvision Dataset instance.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASET_REGISTRY:
        supported = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {supported}"
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform(image_size=image_size, n_channels=n_channels)
    dataset_cls = _DATASET_REGISTRY[dataset_name]

    return dataset_cls(
        root=str(data_dir),
        train=train,
        download=True,
        transform=transform,
    )


def get_dataloader(
    dataset_name: str = "mnist",
    data_dir: str | Path = "data/datasets",
    image_size: int = 28,
    n_channels: int = 1,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    train: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """Get a DataLoader for GAN training.

    Args:
        dataset_name: One of 'mnist', 'fashion_mnist', 'cifar10'.
        data_dir: Root directory for dataset storage.
        image_size: Target spatial resolution.
        n_channels: Number of output channels.
        batch_size: Batch size.
        num_workers: Number of data loading workers (0 for MPS).
        shuffle: Whether to shuffle data.
        train: Whether to load training split.
        drop_last: Drop the last incomplete batch.
        pin_memory: Pin memory for CUDA transfers.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        n_channels=n_channels,
        train=train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def get_dataloader_from_config(config: dict) -> DataLoader:
    """Build a DataLoader directly from a config dictionary.

    Args:
        config: Parsed YAML configuration.

    Returns:
        A PyTorch DataLoader.
    """
    data_cfg = config["data"]
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})

    return get_dataloader(
        dataset_name=data_cfg["dataset_name"],
        data_dir=paths_cfg.get("data_dir", "data/datasets"),
        image_size=data_cfg.get("image_size", 28),
        n_channels=model_cfg.get("n_channels", data_cfg.get("n_channels", 1)),
        batch_size=data_cfg.get("batch_size", 128),
        num_workers=data_cfg.get("num_workers", 0),
        shuffle=data_cfg.get("shuffle", True),
    )
