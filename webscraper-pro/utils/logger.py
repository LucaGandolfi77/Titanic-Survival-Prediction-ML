"""Centralised logging configuration for WebScraper Pro.

Provides a dual-handler logger that writes:
- **INFO+** to the console (coloured via StreamHandler)
- **DEBUG+** to a rotating file ``scraper.log``

Usage::

    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Scraping started")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "scraper.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_INITIALISED: bool = False


def _ensure_log_dir() -> None:
    """Create the log directory if it does not exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


class _ColourFormatter(logging.Formatter):
    """Formatter that emits ANSI colour codes for the console."""

    _COLOURS: dict[int, str] = {
        logging.DEBUG: "\033[90m",      # grey
        logging.INFO: "\033[32m",       # green
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = self._COLOURS.get(record.levelno, "")
        message = super().format(record)
        return f"{colour}{message}{self._RESET}"


def _setup_root_logger() -> None:
    """Configure the root logger once (idempotent)."""
    global _INITIALISED  # noqa: PLW0603
    if _INITIALISED:
        return
    _INITIALISED = True

    _ensure_log_dir()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler (INFO) ──────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        _ColourFormatter("%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
                         datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    # ── File handler (DEBUG, rotating) ──────────────────────────
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising the root logger on first call.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    _setup_root_logger()
    return logging.getLogger(name)
