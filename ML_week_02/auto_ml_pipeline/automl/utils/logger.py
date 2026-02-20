"""Structured logging with loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

# Remove default sink and reconfigure
_loguru_logger.remove()
_loguru_logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> — <level>{message}</level>"
    ),
    colorize=True,
)

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_loguru_logger.add(
    _LOG_DIR / "automl.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
)


def get_logger(name: str = __name__):
    """Return a module-scoped loguru logger."""
    return _loguru_logger.bind(name=name)
