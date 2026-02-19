"""
Model checkpointing — save / load / resume training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    save_dir: str | Path,
    metrics: dict[str, float] | None = None,
    filename: str | None = None,
    keep_last_n: int = 5,
) -> Path:
    """Save a training checkpoint.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        optimizer_g: Generator optimizer.
        optimizer_d: Discriminator optimizer.
        epoch: Current epoch number.
        global_step: Global training step.
        save_dir: Directory to save the checkpoint.
        metrics: Optional dict of metric values to save.
        filename: Optional custom filename; defaults to 'checkpoint_epoch_{epoch}.pt'.
        keep_last_n: Keep only the last N checkpoints (0 = keep all).

    Returns:
        Path to the saved checkpoint file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pt"

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "metrics": metrics or {},
    }

    filepath = save_dir / filename
    torch.save(checkpoint, filepath)

    # Also save a "latest" symlink / copy
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Cleanup old checkpoints
    if keep_last_n > 0:
        _cleanup_old_checkpoints(save_dir, keep_last_n)

    return filepath


def load_checkpoint(
    checkpoint_path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint and restore model/optimizer states.

    Args:
        checkpoint_path: Path to the checkpoint file.
        generator: Generator model (weights will be loaded in-place).
        discriminator: Discriminator model.
        optimizer_g: Optional generator optimizer to restore.
        optimizer_d: Optional discriminator optimizer to restore.
        device: Device to map tensors to.

    Returns:
        Dictionary with 'epoch', 'global_step', and 'metrics'.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])

    if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def find_latest_checkpoint(save_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Looks for 'checkpoint_latest.pt' first, then falls back
    to the highest-numbered checkpoint file.

    Args:
        save_dir: Directory to search.

    Returns:
        Path to the latest checkpoint, or None if none found.
    """
    save_dir = Path(save_dir)
    latest = save_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping only the N most recent."""
    checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt.unlink()
