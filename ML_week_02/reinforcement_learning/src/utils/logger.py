"""Logging: TensorBoard + Python file/console logger.

Provides a unified ``RLLogger`` that can write:
* TensorBoard scalars (rewards, losses, epsilon, eval metrics)
* A human-readable log file
* Console output
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]


class RLLogger:
    """Experiment logger with TensorBoard integration.

    Parameters
    ----------
    experiment_name : str
        Name used for the TensorBoard run sub-folder.
    log_dir : str | Path
        Root directory for TensorBoard logs.
    use_tensorboard : bool
        Whether to write TensorBoard events.
    """

    def __init__(
        self,
        experiment_name: str = "rl_experiment",
        log_dir: str | Path = "outputs/tensorboard",
        use_tensorboard: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self._writer: SummaryWriter | None = None
        if use_tensorboard and SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))

        # Python logger
        self._logger = logging.getLogger(f"RL.{experiment_name}")
        if not self._logger.handlers:
            self._logger.setLevel(logging.INFO)
            fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # Console
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)
            # File
            fh = logging.FileHandler(self.log_dir / f"{experiment_name}.log")
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        loss: float = 0.0,
        epsilon: float = 0.0,
    ) -> None:
        if self._writer:
            self._writer.add_scalar("train/episode_reward", reward, episode)
            self._writer.add_scalar("train/episode_length", length, episode)
            self._writer.add_scalar("train/loss", loss, episode)
            self._writer.add_scalar("train/epsilon", epsilon, episode)

    def log_eval(self, episode: int, metrics: dict[str, float]) -> None:
        self.info(
            f"  [EVAL ep {episode + 1}] "
            f"mean={metrics['mean_reward']:.1f} ± {metrics['std_reward']:.1f}  "
            f"min={metrics['min_reward']:.1f}  max={metrics['max_reward']:.1f}"
        )
        if self._writer:
            for key, val in metrics.items():
                self._writer.add_scalar(f"eval/{key}", val, episode)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
