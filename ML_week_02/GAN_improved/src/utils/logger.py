"""
Logging utilities — TensorBoard + file-based logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class GANLogger:
    """Unified logger for GAN training: TensorBoard + Python logging.

    Usage:
        logger = GANLogger(log_dir="outputs/tensorboard", experiment_name="dcgan-mnist")
        logger.log_scalar("loss/generator", g_loss, step)
        logger.log_images("samples", image_grid, step)
        logger.info("Epoch 10 completed")
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str = "gan_experiment",
        use_tensorboard: bool = True,
        log_to_file: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self._writer: SummaryWriter | None = None
        if use_tensorboard:
            tb_dir = self.log_dir / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(tb_dir))

        # Python logger
        self._logger = logging.getLogger(f"GAN.{experiment_name}")
        self._logger.setLevel(log_level)
        self._logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(fmt)
        self._logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            file_dir = self.log_dir / "logs"
            file_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_dir / f"{experiment_name}.log")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(fmt)
            self._logger.addHandler(file_handler)

    # ── TensorBoard scalars ──────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag."""
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    # ── TensorBoard images ───────────────────────────────────────────────

    def log_images(self, tag: str, images: Tensor, step: int) -> None:
        """Log a batch of images (expects [B, C, H, W] or [C, H, W])."""
        if self._writer is not None:
            if images.dim() == 3:
                self._writer.add_image(tag, images, step)
            else:
                self._writer.add_images(tag, images, step)

    def log_image_grid(self, tag: str, grid: Tensor, step: int) -> None:
        """Log a pre-made image grid [C, H, W]."""
        if self._writer is not None:
            self._writer.add_image(tag, grid, step)

    # ── TensorBoard histograms ───────────────────────────────────────────

    def log_histogram(self, tag: str, values: Tensor, step: int) -> None:
        """Log a histogram of values."""
        if self._writer is not None:
            self._writer.add_histogram(tag, values, step)

    def log_model_gradients(
        self, model: torch.nn.Module, model_name: str, step: int
    ) -> None:
        """Log gradient histograms for all parameters of a model."""
        if self._writer is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._writer.add_histogram(
                    f"{model_name}/gradients/{name}", param.grad, step
                )

    def log_model_weights(
        self, model: torch.nn.Module, model_name: str, step: int
    ) -> None:
        """Log weight histograms for all parameters of a model."""
        if self._writer is None:
            return
        for name, param in model.named_parameters():
            self._writer.add_histogram(
                f"{model_name}/weights/{name}", param.data, step
            )

    # ── TensorBoard graph ────────────────────────────────────────────────

    def log_graph(self, model: torch.nn.Module, input_data: Any) -> None:
        """Log model computational graph."""
        if self._writer is not None:
            self._writer.add_graph(model, input_data)

    # ── Python logging ───────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer and all handlers."""
        if self._writer is not None:
            self._writer.close()
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
