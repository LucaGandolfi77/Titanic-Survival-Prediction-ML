"""Utility helpers for saving image grids and model weights."""

from pathlib import Path

import torch
import torchvision.utils as vutils

from config import IMAGE_SHAPE, SAMPLES_DIR, WEIGHTS_DIR


def save_image_grid(
    images: torch.Tensor,
    epoch: int,
    output_dir: Path = SAMPLES_DIR,
    nrow: int = 8,
) -> Path:
    """Save a grid of generated images as a PNG file.

    Args:
        images: Tensor of shape ``(N, C, H, W)`` with values in [-1, 1].
        epoch: Current training epoch (used in the filename).
        output_dir: Directory where the PNG will be written.
        nrow: Number of images per row in the grid.

    Returns:
        The path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"epoch_{epoch:04d}.png"

    # Rescale from [-1, 1] → [0, 1] for saving
    grid = vutils.make_grid(images.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, filepath)
    print(f"  💾 Saved sample grid → {filepath}")
    return filepath


def reshape_to_image(flat: torch.Tensor) -> torch.Tensor:
    """Reshape a flat tensor back to ``(N, C, H, W)`` image format.

    Args:
        flat: Tensor of shape ``(N, C*H*W)``.

    Returns:
        Tensor of shape ``(N, C, H, W)`` defined by ``IMAGE_SHAPE``.
    """
    return flat.view(-1, *IMAGE_SHAPE)


def save_models(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    output_dir: Path = WEIGHTS_DIR,
) -> None:
    """Persist Generator and Discriminator state dicts to disk.

    Args:
        generator: Trained Generator instance.
        discriminator: Trained Discriminator instance.
        output_dir: Directory where ``.pth`` files will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    g_path = output_dir / "generator.pth"
    d_path = output_dir / "discriminator.pth"
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
    print(f"  💾 Generator  weights → {g_path}")
    print(f"  💾 Discriminator weights → {d_path}")
