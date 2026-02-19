"""
logging.py – Structured logging with loguru.

Provides a pre-configured logger for the whole project.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# Remove default handler; add custom ones
logger.remove()

# Console – colourful, INFO+
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# File – JSON, DEBUG+
logger.add(
    _LOG_DIR / "defect_detection.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    serialize=True,
)


def get_logger(name: str = "defect_detection"):
    """Return the shared loguru logger (name is informational only)."""
    return logger
