"""
logging_utils.py — Per-VM log files and a global lifecycle event log.

Each VM gets its own rotating log file under ``<state_dir>/logs/<name>.log``.
A global event log records lifecycle transitions (start, stop, crash, …) for
auditing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level log directory setup
# ---------------------------------------------------------------------------

def _ensure_log_dir(state_dir: Path) -> Path:
    """Create and return ``<state_dir>/logs/``.

    Args:
        state_dir: Base state directory.

    Returns:
        The ``logs/`` subdirectory.
    """
    d = state_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Per-VM log tail
# ---------------------------------------------------------------------------

def tail_log(
    name: str,
    state_dir: Path,
    lines: int = 20,
) -> List[str]:
    """Return the last *lines* lines from a VM's log file.

    Args:
        name:      VM name.
        state_dir: Base state directory.
        lines:     Number of trailing lines to return.

    Returns:
        A list of strings (may be empty if the log doesn't exist).
    """
    log_path = _ensure_log_dir(state_dir) / f"{name}.log"
    if not log_path.exists():
        return []

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        return [l.rstrip("\n") for l in all_lines[-lines:]]
    except OSError as exc:
        logger.warning("Cannot read log for '%s': %s", name, exc)
        return []


def log_path_for(name: str, state_dir: Path) -> Path:
    """Return the path to a VM's log file (may not exist yet).

    Args:
        name:      VM name.
        state_dir: Base directory.

    Returns:
        A ``Path`` object.
    """
    return _ensure_log_dir(state_dir) / f"{name}.log"


# ---------------------------------------------------------------------------
# Global lifecycle event log
# ---------------------------------------------------------------------------

_EVENT_LOG_NAME = "events.log"


def log_event(
    state_dir: Path,
    vm_name: str,
    event: str,
    detail: str = "",
) -> None:
    """Append a one-line event record to the global event log.

    Args:
        state_dir: Base state directory.
        vm_name:   VM that the event pertains to.
        event:     Event tag (``"START"``, ``"STOP"``, ``"CRASH"``, …).
        detail:    Optional human-readable detail string.
    """
    log_dir = _ensure_log_dir(state_dir)
    event_path = log_dir / _EVENT_LOG_NAME
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts}  {event:<8s}  {vm_name}"
    if detail:
        line += f"  {detail}"
    line += "\n"
    try:
        with event_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        logger.warning("Cannot write event log: %s", exc)


# ---------------------------------------------------------------------------
# Python logging configuration
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI use.

    Args:
        verbose: If ``True``, set level to ``DEBUG``; otherwise ``WARNING``.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
