"""
Visualization utilities for GAN outputs.

Includes:
- Image grid generation
- Latent space interpolation (linear + spherical)
- Class-conditioned sample grids
- Training progress animation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid, save_image

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GANVisualizer:
    """Generate and save visualizations for GAN outputs."""

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample Grids ─────────────────────────────────────────────────────

    @staticmethod
    def make_image_grid(
        images: Tensor,
        nrow: int = 8,
        normalize: bool = True,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> Tensor:
        """Create an image grid from a batch of images.

        Args:
            images: Image tensor [B, C, H, W].
            nrow: Number of images per row.
            normalize: Whether to normalize to [0, 1].
            value_range: Range of input values for normalization.

        Returns:
            Grid tensor [C, grid_H, grid_W].
        """
        return make_grid(
            images, nrow=nrow, normalize=normalize, value_range=value_range, padding=2
        )

    def save_sample_grid(
        self,
        images: Tensor,
        filename: str,
        nrow: int = 8,
        subfolder: str = "samples",
    ) -> Path:
        """Save a grid of generated images.

        Args:
            images: Image tensor [B, C, H, W] in [-1, 1].
            filename: Output filename (e.g., 'epoch_010.png').
            nrow: Number of images per row.
            subfolder: Subfolder within output_dir.

        Returns:
            Path to the saved image.
        """
        save_dir = self.output_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        save_image(
            images, str(filepath), nrow=nrow, normalize=True, value_range=(-1, 1)
        )
        return filepath

    # ── Latent Space Interpolation ───────────────────────────────────────

    @staticmethod
    def linear_interpolation(z1: Tensor, z2: Tensor, n_steps: int = 10) -> Tensor:
        """Linear interpolation between two latent vectors.

        Args:
            z1: Start latent vector [latent_dim].
            z2: End latent vector [latent_dim].
            n_steps: Number of interpolation steps.

        Returns:
            Interpolated vectors [n_steps, latent_dim].
        """
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)
        return torch.stack([z1 * (1 - a) + z2 * a for a in alphas])

    @staticmethod
    def spherical_interpolation(z1: Tensor, z2: Tensor, n_steps: int = 10) -> Tensor:
        """Spherical linear interpolation (slerp) — better for high-dim latent spaces.

        Args:
            z1: Start latent vector [latent_dim].
            z2: End latent vector [latent_dim].
            n_steps: Number of interpolation steps.

        Returns:
            Interpolated vectors [n_steps, latent_dim].
        """
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        omega = torch.acos(torch.clamp(torch.dot(z1_norm, z2_norm), -1.0, 1.0))

        if omega.abs() < 1e-6:
            # Vectors are nearly parallel — fall back to linear
            return GANVisualizer.linear_interpolation(z1, z2, n_steps)

        sin_omega = torch.sin(omega)
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)

        return torch.stack([
            (torch.sin((1 - a) * omega) / sin_omega) * z1 +
            (torch.sin(a * omega) / sin_omega) * z2
            for a in alphas
        ])

    @torch.no_grad()
    def save_interpolation(
        self,
        generator: nn.Module,
        latent_dim: int,
        device: torch.device,
        n_pairs: int = 5,
        n_steps: int = 10,
        method: str = "slerp",
        filename: str = "interpolation.png",
        labels: Tensor | None = None,
    ) -> Path:
        """Generate and save latent space interpolation visualization.

        Args:
            generator: Trained generator model.
            latent_dim: Dimension of latent space.
            device: Device to run on.
            n_pairs: Number of interpolation pairs (rows).
            n_steps: Steps per interpolation.
            method: 'linear' or 'slerp'.
            filename: Output filename.
            labels: Optional labels for conditional generation [n_pairs].

        Returns:
            Path to the saved image.
        """
        generator.eval()
        interp_fn = (
            self.spherical_interpolation if method == "slerp"
            else self.linear_interpolation
        )

        all_images = []
        for i in range(n_pairs):
            z1 = torch.randn(latent_dim, device=device)
            z2 = torch.randn(latent_dim, device=device)
            z_interp = interp_fn(z1, z2, n_steps)

            if labels is not None:
                label = labels[i].unsqueeze(0).expand(n_steps)
                images = generator(z_interp, label)
            else:
                images = generator(z_interp)

            all_images.append(images)

        all_images_tensor = torch.cat(all_images, dim=0)

        save_dir = self.output_dir / "interpolations"
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        save_image(
            all_images_tensor,
            str(filepath),
            nrow=n_steps,
            normalize=True,
            value_range=(-1, 1),
        )
        return filepath

    # ── Class-Conditioned Grid (for cGAN) ────────────────────────────────

    @torch.no_grad()
    def save_conditional_grid(
        self,
        generator: nn.Module,
        latent_dim: int,
        n_classes: int,
        device: torch.device,
        n_samples_per_class: int = 8,
        filename: str = "conditional_grid.png",
    ) -> Path:
        """Generate a grid of class-conditioned samples.

        Each row corresponds to a different class.

        Args:
            generator: Trained conditional generator.
            latent_dim: Dimension of latent space.
            n_classes: Number of classes.
            device: Device to run on.
            n_samples_per_class: Samples per class (columns).
            filename: Output filename.

        Returns:
            Path to the saved image.
        """
        generator.eval()
        all_images = []

        for class_idx in range(n_classes):
            z = torch.randn(n_samples_per_class, latent_dim, device=device)
            labels = torch.full(
                (n_samples_per_class,), class_idx, dtype=torch.long, device=device
            )
            images = generator(z, labels)
            all_images.append(images)

        all_images_tensor = torch.cat(all_images, dim=0)

        save_dir = self.output_dir / "samples"
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        save_image(
            all_images_tensor,
            str(filepath),
            nrow=n_samples_per_class,
            normalize=True,
            value_range=(-1, 1),
        )
        return filepath

    # ── Training Progress Plot ───────────────────────────────────────────

    @staticmethod
    def plot_training_curves(
        g_losses: list[float],
        d_losses: list[float],
        save_path: str | Path,
        fid_scores: list[tuple[int, float]] | None = None,
    ) -> None:
        """Plot generator and discriminator loss curves.

        Args:
            g_losses: Generator losses per step.
            d_losses: Discriminator losses per step.
            save_path: Path to save the plot.
            fid_scores: Optional list of (epoch, fid) tuples.
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 2 if fid_scores else 1, figsize=(14, 5))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Loss curves
        ax = axes[0]
        ax.plot(g_losses, label="Generator", alpha=0.7)
        ax.plot(d_losses, label="Discriminator", alpha=0.7)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("GAN Training Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # FID curve
        if fid_scores and len(axes) > 1:
            ax2 = axes[1]
            epochs, fids = zip(*fid_scores)
            ax2.plot(epochs, fids, "o-", color="green")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("FID Score")
            ax2.set_title("FID Score Over Training")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
