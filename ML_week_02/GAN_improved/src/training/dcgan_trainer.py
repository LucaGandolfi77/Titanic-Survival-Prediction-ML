"""
Stage 2 — DCGAN Trainer.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..models.dcgan import build_dcgan
from .trainer import BaseTrainer


class DCGANTrainer(BaseTrainer):
    """Trainer for the Deep Convolutional GAN (Stage 2)."""

    def _build_models(self) -> None:
        """Initialize DCGAN models, optimizers, and loss."""
        self.generator, self.discriminator = build_dcgan(self.config)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        train_cfg = self.config["training"]
        self.optimizer_g = self._build_optimizer(
            self.generator.parameters(), train_cfg["optimizer_g"]
        )
        self.optimizer_d = self._build_optimizer(
            self.discriminator.parameters(), train_cfg["optimizer_d"]
        )
        self.loss_fn = self._get_loss_fn()

    def _train_discriminator_step(
        self, real_images: Tensor, real_labels: Tensor | None = None,
    ) -> float:
        """Train DCGAN discriminator."""
        self.optimizer_d.zero_grad()
        batch_size = real_images.size(0)

        # Real images
        real_target = self._get_real_labels(batch_size)
        real_output = self.discriminator(real_images)
        d_loss_real = self.loss_fn(real_output, real_target)

        # Fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z).detach()
        fake_target = self._get_fake_labels(batch_size)
        fake_output = self.discriminator(fake_images)
        d_loss_fake = self.loss_fn(fake_output, fake_target)

        d_loss = d_loss_real + d_loss_fake

        if self.use_gradient_penalty:
            gp = self._compute_gradient_penalty(real_images, fake_images)
            d_loss = d_loss + self.gp_lambda * gp

        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def _train_generator_step(
        self, batch_size: int, real_labels: Tensor | None = None,
    ) -> float:
        """Train DCGAN generator (non-saturating loss)."""
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images)

        real_target = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.loss_fn(fake_output, real_target)

        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    @torch.no_grad()
    def _generate_samples(self, n_samples: int) -> Tensor:
        """Generate samples using fixed noise."""
        self.generator.eval()
        noise = self.fixed_noise[:n_samples]
        return self.generator(noise)
