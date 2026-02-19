"""
Stage 3 — Conditional GAN (cGAN with DCGAN backbone).

Class-conditioned generation using:
  • Embedding + projection in Generator (concat to latent)
  • Projection Discriminator (Miyato & Koyama, 2018)

Allows generating specific digit classes on demand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import apply_spectral_norm, weights_init_normal


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN Generator.

    The class label is embedded and concatenated with the latent vector z
    before being fed into the transposed convolution stack.

    Architecture (64×64 output):
        [z; embed(y)] → ConvTranspose2d stack → image
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_channels: int = 1,
        n_classes: int = 10,
        ngf: int = 64,
        embedding_dim: int = 50,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Class embedding
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        input_dim = latent_dim + embedding_dim

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
            # [B, input_dim, 1, 1] → [B, ngf*8, 4, 4]
            *_block(input_dim, ngf * 8, kernel=4, stride=1, padding=0),
            # → [B, ngf*4, 8, 8]
            *_block(ngf * 8, ngf * 4),
            # → [B, ngf*2, 16, 16]
            *_block(ngf * 4, ngf * 2),
            # → [B, ngf, 32, 32]
            *_block(ngf * 2, ngf),
            # → [B, nc, 64, 64]
            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

        self.apply(weights_init_normal)

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """Generate class-conditioned images.

        Args:
            z: Latent vectors [B, latent_dim]
            labels: Class labels [B] (integer indices)

        Returns:
            Generated images [B, C, 64, 64]
        """
        label_emb = self.label_embedding(labels)           # [B, embedding_dim]
        gen_input = torch.cat([z, label_emb], dim=1)       # [B, latent_dim + embedding_dim]
        gen_input = gen_input.unsqueeze(-1).unsqueeze(-1)   # [B, D, 1, 1]
        return self.net(gen_input)


class ConditionalDiscriminator(nn.Module):
    """
    Projection Discriminator for Conditional GAN.

    Uses the projection technique from Miyato & Koyama (2018):
    the class embedding is projected (inner product) with the penultimate
    feature vector, rather than naively concatenated.

    Architecture (64×64 input):
        image → Conv2d stack → features
        output = linear(features) + <embed(y), features>   (projection)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 10,
        ndf: int = 64,
        use_batch_norm: bool = False,
        use_spectral_norm: bool = True,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

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

        # Feature extractor: [B, nc, 64, 64] → [B, ndf*8, 4, 4]
        self.features = nn.Sequential(
            *_block(n_channels, ndf, bn=False),
            *_block(ndf, ndf * 2),
            *_block(ndf * 2, ndf * 4),
            *_block(ndf * 4, ndf * 8),
        )

        # Global average pooling will give [B, ndf*8]
        feature_dim = ndf * 8

        # Unconditional logit
        self.linear = nn.Linear(feature_dim, 1)

        # Projection: class embedding for inner product
        self.label_embedding = nn.Embedding(n_classes, feature_dim)

        self.apply(weights_init_normal)

        if use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, img: Tensor, labels: Tensor) -> Tensor:
        """Classify images as real/fake, conditioned on class labels.

        Uses projection discriminator:
            output = linear(h) + <embed(y), h>

        Args:
            img: Images [B, C, 64, 64]
            labels: Class labels [B] (integer indices)

        Returns:
            Logits [B, 1]
        """
        h = self.features(img)                              # [B, ndf*8, 4, 4]
        h = h.mean(dim=[2, 3])                              # GAP → [B, ndf*8]

        out = self.linear(h)                                # [B, 1] unconditional
        label_emb = self.label_embedding(labels)            # [B, ndf*8]
        # Projection: inner product → [B, 1]
        proj = (h * label_emb).sum(dim=1, keepdim=True)
        return out + proj


def build_conditional_gan(
    config: dict,
) -> tuple[ConditionalGenerator, ConditionalDiscriminator]:
    """Factory function to build Conditional GAN from config dict."""
    model_cfg = config["model"]
    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    generator = ConditionalGenerator(
        latent_dim=model_cfg["latent_dim"],
        n_channels=model_cfg["n_channels"],
        n_classes=model_cfg.get("n_classes", 10),
        ngf=model_cfg.get("ngf", 64),
        embedding_dim=model_cfg.get("embedding_dim", 50),
        use_batch_norm=gen_cfg.get("use_batch_norm", True),
    )

    discriminator = ConditionalDiscriminator(
        n_channels=model_cfg["n_channels"],
        n_classes=model_cfg.get("n_classes", 10),
        ndf=model_cfg.get("ndf", 64),
        use_batch_norm=disc_cfg.get("use_batch_norm", False),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", True),
        leaky_slope=disc_cfg.get("leaky_slope", 0.2),
    )

    return generator, discriminator
