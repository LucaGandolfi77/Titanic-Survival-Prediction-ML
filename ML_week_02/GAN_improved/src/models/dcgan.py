"""
Stage 2 — DCGAN (Deep Convolutional GAN).

Follows Radford et al. (2016) architectural guidelines:
  • Replace pooling with strided convolutions (D) and fractional-strided convolutions (G)
  • Use BatchNorm in both G and D (except D output and G input)
  • Remove fully connected layers for deeper architectures
  • ReLU in G (except output: Tanh), LeakyReLU in D

With modern additions: Spectral Normalization in D.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .layers import apply_spectral_norm, weights_init_normal


class DCGANGenerator(nn.Module):
    """
    Convolutional Generator for 64×64 images.

    Architecture (for ngf=64, latent_dim=100):
        z [B, 100, 1, 1]
        → ConvTranspose2d → [B, ngf*8, 4, 4]    # Project & reshape
        → ConvTranspose2d → [B, ngf*4, 8, 8]
        → ConvTranspose2d → [B, ngf*2, 16, 16]
        → ConvTranspose2d → [B, ngf,   32, 32]
        → ConvTranspose2d → [B, nc,    64, 64]   # Output: Tanh
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_channels: int = 1,
        ngf: int = 64,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        def _block(
            in_c: int,
            out_c: int,
            kernel: int = 4,
            stride: int = 2,
            padding: int = 1,
            bn: bool = True,
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=not bn)
            ]
            if bn and use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.net = nn.Sequential(
            # Input: z [B, latent_dim, 1, 1] → [B, ngf*8, 4, 4]
            *_block(latent_dim, ngf * 8, kernel=4, stride=1, padding=0),
            # [B, ngf*8, 4, 4] → [B, ngf*4, 8, 8]
            *_block(ngf * 8, ngf * 4),
            # [B, ngf*4, 8, 8] → [B, ngf*2, 16, 16]
            *_block(ngf * 4, ngf * 2),
            # [B, ngf*2, 16, 16] → [B, ngf, 32, 32]
            *_block(ngf * 2, ngf),
            # [B, ngf, 32, 32] → [B, nc, 64, 64]
            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

        self.apply(weights_init_normal)

    def forward(self, z: Tensor) -> Tensor:
        """Generate images from latent vectors.

        Args:
            z: Latent vectors [B, latent_dim] or [B, latent_dim, 1, 1]

        Returns:
            Generated images [B, C, 64, 64]
        """
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """
    Convolutional Discriminator for 64×64 images.

    Architecture (for ndf=64):
        [B, nc, 64, 64]
        → Conv2d → [B, ndf,   32, 32]   # No BatchNorm on first layer
        → Conv2d → [B, ndf*2, 16, 16]
        → Conv2d → [B, ndf*4, 8, 8]
        → Conv2d → [B, ndf*8, 4, 4]
        → Conv2d → [B, 1,     1, 1]     # Output logit
    """

    def __init__(
        self,
        n_channels: int = 1,
        ndf: int = 64,
        use_batch_norm: bool = False,
        use_spectral_norm: bool = True,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()

        def _block(
            in_c: int,
            out_c: int,
            kernel: int = 4,
            stride: int = 2,
            padding: int = 1,
            bn: bool = True,
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not (bn and use_batch_norm))
            ]
            if bn and use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            return layers

        self.net = nn.Sequential(
            # [B, nc, 64, 64] → [B, ndf, 32, 32]  (no BN on first layer)
            *_block(n_channels, ndf, bn=False),
            # [B, ndf, 32, 32] → [B, ndf*2, 16, 16]
            *_block(ndf, ndf * 2),
            # [B, ndf*2, 16, 16] → [B, ndf*4, 8, 8]
            *_block(ndf * 2, ndf * 4),
            # [B, ndf*4, 8, 8] → [B, ndf*8, 4, 4]
            *_block(ndf * 4, ndf * 8),
            # [B, ndf*8, 4, 4] → [B, 1, 1, 1]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            # No Sigmoid — BCEWithLogitsLoss
        )

        self.apply(weights_init_normal)

        if use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, img: Tensor) -> Tensor:
        """Classify images as real/fake.

        Args:
            img: Images [B, C, 64, 64]

        Returns:
            Logits [B, 1]
        """
        return self.net(img).view(-1, 1)


def build_dcgan(config: dict) -> tuple[DCGANGenerator, DCGANDiscriminator]:
    """Factory function to build DCGAN from config dict."""
    model_cfg = config["model"]
    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    generator = DCGANGenerator(
        latent_dim=model_cfg["latent_dim"],
        n_channels=model_cfg["n_channels"],
        ngf=model_cfg.get("ngf", 64),
        use_batch_norm=gen_cfg.get("use_batch_norm", True),
    )

    discriminator = DCGANDiscriminator(
        n_channels=model_cfg["n_channels"],
        ndf=model_cfg.get("ndf", 64),
        use_batch_norm=disc_cfg.get("use_batch_norm", False),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", True),
        leaky_slope=disc_cfg.get("leaky_slope", 0.2),
    )

    return generator, discriminator
