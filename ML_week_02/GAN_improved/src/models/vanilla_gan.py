"""
Stage 1 — Vanilla GAN (MLP-based Generator + Discriminator).

Architecture follows Goodfellow et al. (2014) with modern stability
improvements: spectral normalization, LeakyReLU, dropout in D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import apply_spectral_norm, weights_init_normal


class VanillaGenerator(nn.Module):
    """
    MLP Generator: maps latent vector z → flattened image.

    Architecture:
        z (latent_dim) → 256 → 512 → 1024 → image_dim
        BatchNorm + LeakyReLU between layers, Tanh output.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        image_shape: tuple[int, ...] = (1, 28, 28),
        hidden_dims: list[int] | None = None,
        use_batch_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.image_dim = 1
        for s in image_shape:
            self.image_dim *= s

        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, self.image_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, z: Tensor) -> Tensor:
        """Generate images from latent vectors.

        Args:
            z: Latent vectors [B, latent_dim]

        Returns:
            Generated images [B, C, H, W]
        """
        out = self.net(z)
        return out.view(out.size(0), *self.image_shape)


class VanillaDiscriminator(nn.Module):
    """
    MLP Discriminator: maps flattened image → real/fake probability.

    Architecture:
        image_dim → 512 → 256 → 1
        LeakyReLU + Dropout, optional spectral normalization.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...] = (1, 28, 28),
        hidden_dims: list[int] | None = None,
        use_spectral_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.image_dim = 1
        for s in image_shape:
            self.image_dim *= s

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[nn.Module] = []
        in_dim = self.image_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        # No Sigmoid here — applied in loss (BCEWithLogitsLoss) for stability

        self.net = nn.Sequential(*layers)
        self.apply(weights_init_normal)

        if use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, img: Tensor) -> Tensor:
        """Classify images as real/fake.

        Args:
            img: Images [B, C, H, W]

        Returns:
            Logits [B, 1]
        """
        flat = img.view(img.size(0), -1)
        return self.net(flat)


def build_vanilla_gan(config: dict) -> tuple[VanillaGenerator, VanillaDiscriminator]:
    """Factory function to build Vanilla GAN from config dict."""
    model_cfg = config["model"]
    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    image_shape = (model_cfg["n_channels"], model_cfg["image_size"], model_cfg["image_size"])

    generator = VanillaGenerator(
        latent_dim=model_cfg["latent_dim"],
        image_shape=image_shape,
        hidden_dims=gen_cfg.get("hidden_dims"),
        use_batch_norm=gen_cfg.get("use_batch_norm", True),
        leaky_slope=gen_cfg.get("leaky_slope", 0.2),
        dropout=gen_cfg.get("dropout", 0.0),
    )

    discriminator = VanillaDiscriminator(
        image_shape=image_shape,
        hidden_dims=disc_cfg.get("hidden_dims"),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", True),
        leaky_slope=disc_cfg.get("leaky_slope", 0.2),
        dropout=disc_cfg.get("dropout", 0.3),
    )

    return generator, discriminator
