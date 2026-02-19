"""
Stage 3 — Conditional GAN Trainer.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..models.conditional_gan import build_conditional_gan
from .trainer import BaseTrainer


class ConditionalGANTrainer(BaseTrainer):
    """Trainer for the Conditional GAN with projection discriminator (Stage 3)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.n_classes = config["model"].get("n_classes", 10)

        # Fixed labels for visualization: one of each class
        n_per_class = 64 // self.n_classes
        self.fixed_labels = torch.arange(self.n_classes, device=self.device).repeat(
            n_per_class + 1
        )[:64]

    def _build_models(self) -> None:
        """Initialize Conditional GAN models, optimizers, and loss."""
        self.generator, self.discriminator = build_conditional_gan(self.config)
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
        """Train conditional discriminator with real class labels."""
        assert real_labels is not None, "Conditional GAN requires class labels"
        self.optimizer_d.zero_grad()
        batch_size = real_images.size(0)

        # Real images with real labels
        real_target = self._get_real_labels(batch_size)
        real_output = self.discriminator(real_images, real_labels)
        d_loss_real = self.loss_fn(real_output, real_target)

        # Fake images with real labels (G tries to fool D for given class)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z, real_labels).detach()
        fake_target = self._get_fake_labels(batch_size)
        fake_output = self.discriminator(fake_images, real_labels)
        d_loss_fake = self.loss_fn(fake_output, fake_target)

        d_loss = d_loss_real + d_loss_fake

        if self.use_gradient_penalty:
            # For cGAN, we pass labels through a lambda
            gp = self._compute_gradient_penalty(
                real_images,
                fake_images,
                discriminator_fn=lambda x: self.discriminator(x, real_labels),
            )
            d_loss = d_loss + self.gp_lambda * gp

        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def _train_generator_step(
        self, batch_size: int, real_labels: Tensor | None = None,
    ) -> float:
        """Train conditional generator: generate for given class labels."""
        assert real_labels is not None, "Conditional GAN requires class labels"
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z, real_labels)
        fake_output = self.discriminator(fake_images, real_labels)

        real_target = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.loss_fn(fake_output, real_target)

        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    @torch.no_grad()
    def _generate_samples(self, n_samples: int) -> Tensor:
        """Generate class-conditioned samples."""
        self.generator.eval()
        noise = self.fixed_noise[:n_samples]
        labels = self.fixed_labels[:n_samples]
        return self.generator(noise, labels)

    def _save_epoch_samples(self, epoch: int) -> None:
        """Override to also save class-conditioned grids."""
        # Standard samples (mixed classes)
        super()._save_epoch_samples(epoch)

        # Class-conditioned grid
        self.visualizer.save_conditional_grid(
            generator=self.generator,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
            device=self.device,
            n_samples_per_class=8,
            filename=f"conditional_grid_epoch_{epoch:04d}.png",
        )
