"""
Base GAN Trainer — abstract class encapsulating the training loop.

All stage-specific trainers inherit from this class and override:
  - _build_models()
  - _train_discriminator_step()
  - _train_generator_step()
  - _generate_samples()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.visualization import GANVisualizer
from ..utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..utils.logger import GANLogger


class BaseTrainer(ABC):
    """Abstract base class for GAN training.

    Handles:
    - Training loop orchestration
    - Loss computation with label smoothing / noise
    - Checkpoint management (save/load/resume)
    - TensorBoard logging
    - Periodic sample generation
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device(config["experiment"]["device"])

        # Training params
        train_cfg = config["training"]
        self.n_epochs = train_cfg["n_epochs"]
        self.d_steps_per_g_step = train_cfg.get("d_steps_per_g_step", 1)
        self.label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.label_noise = train_cfg.get("label_noise", 0.0)
        self.loss_type = train_cfg.get("loss_type", "bce")
        self.use_gradient_penalty = train_cfg.get("gradient_penalty", False)
        self.gp_lambda = train_cfg.get("gradient_penalty_lambda", 10.0)

        # Evaluation params
        eval_cfg = config.get("evaluation", {})
        self.sample_interval = eval_cfg.get("sample_interval", 5)
        self.fid_interval = eval_cfg.get("fid_interval", 20)
        self.n_fid_samples = eval_cfg.get("n_fid_samples", 5000)

        # Logging params
        log_cfg = config.get("logging", {})
        self.save_interval = log_cfg.get("save_interval", 10)
        self.log_every_n_steps = log_cfg.get("log_every_n_steps", 50)

        # Paths
        paths_cfg = config.get("paths", {})
        self.models_dir = Path(paths_cfg.get("models_dir", "outputs/models"))
        self.samples_dir = Path(paths_cfg.get("samples_dir", "outputs/samples"))

        # Initialize components
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.optimizer_g: torch.optim.Optimizer | None = None
        self.optimizer_d: torch.optim.Optimizer | None = None
        self.loss_fn: nn.Module | None = None

        # Logger & Visualizer
        self.logger = GANLogger(
            log_dir=log_cfg.get("log_dir", "outputs/tensorboard"),
            experiment_name=config["experiment"]["name"],
            use_tensorboard=log_cfg.get("tensorboard", True),
        )
        self.visualizer = GANVisualizer(output_dir=str(self.samples_dir.parent))

        # Track training state
        self.global_step = 0
        self.start_epoch = 0
        self.g_losses: list[float] = []
        self.d_losses: list[float] = []
        self.fid_scores: list[tuple[int, float]] = []

        # Fixed noise for consistent sample visualization
        self.latent_dim = config["model"]["latent_dim"]
        self.fixed_noise = torch.randn(64, self.latent_dim, device=self.device)

    # ── Abstract methods (must be overridden) ────────────────────────────

    @abstractmethod
    def _build_models(self) -> None:
        """Initialize generator, discriminator, optimizers, and loss."""
        ...

    @abstractmethod
    def _train_discriminator_step(
        self, real_images: Tensor, real_labels: Tensor | None = None,
    ) -> float:
        """One discriminator training step. Returns D loss."""
        ...

    @abstractmethod
    def _train_generator_step(
        self, batch_size: int, real_labels: Tensor | None = None,
    ) -> float:
        """One generator training step. Returns G loss."""
        ...

    @abstractmethod
    def _generate_samples(self, n_samples: int) -> Tensor:
        """Generate n_samples images using the generator."""
        ...

    # ── Loss utilities ───────────────────────────────────────────────────

    def _get_loss_fn(self) -> nn.Module:
        """Get the appropriate loss function."""
        if self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.loss_type == "mse":
            return nn.MSELoss()
        else:
            # For wasserstein / hinge, losses are computed inline
            return nn.BCEWithLogitsLoss()

    def _get_real_labels(self, batch_size: int) -> Tensor:
        """Get labels for real data with optional label smoothing + noise."""
        labels = torch.ones(batch_size, 1, device=self.device)
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing)
        if self.label_noise > 0:
            labels += self.label_noise * torch.randn_like(labels)
            labels = labels.clamp(0, 1)
        return labels

    def _get_fake_labels(self, batch_size: int) -> Tensor:
        """Get labels for fake data with optional noise."""
        labels = torch.zeros(batch_size, 1, device=self.device)
        if self.label_noise > 0:
            labels += self.label_noise * torch.randn_like(labels).abs()
            labels = labels.clamp(0, 1)
        return labels

    def _compute_gradient_penalty(
        self, real_data: Tensor, fake_data: Tensor,
        discriminator_fn: Any = None,
    ) -> Tensor:
        """Compute gradient penalty (WGAN-GP style).

        Args:
            real_data: Real images [B, C, H, W].
            fake_data: Generated images [B, C, H, W].
            discriminator_fn: Callable discriminator.

        Returns:
            Gradient penalty scalar.
        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        disc_fn = discriminator_fn or self.discriminator
        d_interpolated = disc_fn(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ── Build optimizer ──────────────────────────────────────────────────

    def _build_optimizer(
        self, params: Any, opt_config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Build an optimizer from config."""
        opt_type = opt_config.get("type", "adam").lower()
        lr = opt_config.get("lr", 0.0002)
        betas = tuple(opt_config.get("betas", [0.5, 0.999]))

        if opt_type == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas)
        elif opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt_type == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    # ── Training loop ────────────────────────────────────────────────────

    def train(self, dataloader: DataLoader, resume: bool = False) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            dataloader: Training data loader.
            resume: Whether to resume from the latest checkpoint.

        Returns:
            Dictionary with training history and final metrics.
        """
        self._build_models()

        if resume:
            self._try_resume()

        self.logger.info(
            f"Starting training: {self.config['experiment']['name']} "
            f"on {self.device} for {self.n_epochs} epochs"
        )
        self.logger.info(
            f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}"
        )
        self.logger.info(
            f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}"
        )

        start_time = time.time()

        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            self.generator.train()
            self.discriminator.train()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.n_epochs}")
            for batch_idx, (real_images, labels) in enumerate(pbar):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_images.size(0)

                # Train Discriminator (possibly multiple steps)
                d_loss = 0.0
                for _ in range(self.d_steps_per_g_step):
                    d_loss = self._train_discriminator_step(real_images, labels)

                # Train Generator
                g_loss = self._train_generator_step(batch_size, labels)

                # Track losses
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                n_batches += 1
                self.global_step += 1

                # Log to TensorBoard
                if self.global_step % self.log_every_n_steps == 0:
                    self.logger.log_scalar("loss/generator", g_loss, self.global_step)
                    self.logger.log_scalar("loss/discriminator", d_loss, self.global_step)

                # Update progress bar
                pbar.set_postfix({"G_loss": f"{g_loss:.4f}", "D_loss": f"{d_loss:.4f}"})

            # Epoch averages
            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            self.logger.log_scalar("loss/generator_epoch", avg_g, epoch)
            self.logger.log_scalar("loss/discriminator_epoch", avg_d, epoch)
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.n_epochs}] "
                f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f}"
            )

            # Generate samples
            if (epoch + 1) % self.sample_interval == 0:
                self._save_epoch_samples(epoch + 1)

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                save_checkpoint(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    optimizer_g=self.optimizer_g,
                    optimizer_d=self.optimizer_d,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    save_dir=self.models_dir,
                    metrics={"g_loss": avg_g, "d_loss": avg_d},
                )
                self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed / 60:.1f} minutes")

        # Save final checkpoint
        save_checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            epoch=self.n_epochs,
            global_step=self.global_step,
            save_dir=self.models_dir,
            filename="checkpoint_final.pt",
        )

        # Plot training curves
        self.visualizer.plot_training_curves(
            self.g_losses,
            self.d_losses,
            save_path=self.samples_dir.parent / "training_curves.png",
            fid_scores=self.fid_scores if self.fid_scores else None,
        )

        self.logger.flush()
        self.logger.close()

        return {
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "fid_scores": self.fid_scores,
            "total_steps": self.global_step,
            "training_time_minutes": elapsed / 60,
        }

    def _save_epoch_samples(self, epoch: int) -> None:
        """Generate and save sample images for the current epoch."""
        self.generator.eval()
        with torch.no_grad():
            samples = self._generate_samples(64)
        grid = self.visualizer.make_image_grid(samples)
        self.logger.log_image_grid(
            f"samples/{self.config['experiment']['stage']}", grid, epoch
        )
        stage = self.config["experiment"]["stage"]
        self.visualizer.save_sample_grid(
            samples,
            filename=f"{stage}_epoch_{epoch:04d}.png",
            subfolder="samples",
        )

    def _try_resume(self) -> None:
        """Try to resume from the latest checkpoint."""
        ckpt_path = find_latest_checkpoint(self.models_dir)
        if ckpt_path is not None:
            self.logger.info(f"Resuming from checkpoint: {ckpt_path}")
            info = load_checkpoint(
                ckpt_path,
                self.generator,
                self.discriminator,
                self.optimizer_g,
                self.optimizer_d,
                device=self.device,
            )
            self.start_epoch = info["epoch"]
            self.global_step = info["global_step"]
            self.logger.info(f"Resumed at epoch {self.start_epoch}, step {self.global_step}")
        else:
            self.logger.info("No checkpoint found — starting from scratch")
