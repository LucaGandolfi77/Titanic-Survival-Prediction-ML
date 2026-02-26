"""
repository.py – Persistence layer for RapidMiner‑Lite.

Handles:
  - Saving / loading Process files (.rmp — JSON)
  - Storing / retrieving ExampleSets and models (pickle)
  - Managing sample processes shipped with the app
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.operator_base import _serialise_val
from engine.process_runner import Process

logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────

APP_DIR = Path.home() / ".rapidminerlite"
PROCESSES_DIR = APP_DIR / "processes"
RESULTS_DIR = APP_DIR / "results"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_processes"


def _ensure_dirs() -> None:
    PROCESSES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Process I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_process(process: Process, path: Optional[Path] = None) -> Path:
    """Persist a Process as JSON (.rmp).

    Args:
        process: the process to save.
        path: explicit file path; if None, saves under PROCESSES_DIR.

    Returns:
        The path the file was written to.
    """
    _ensure_dirs()
    if path is None:
        safe_name = process.name.replace(" ", "_").replace("/", "_")
        path = PROCESSES_DIR / f"{safe_name}.rmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = process.to_dict()
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Saved process: %s", path)
    return path


def load_process(path: Path) -> Process:
    """Load a Process from a JSON .rmp file."""
    if not path.exists():
        raise FileNotFoundError(f"Process file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Process.from_dict(data)


def list_saved_processes() -> List[Path]:
    """List all .rmp files in PROCESSES_DIR."""
    _ensure_dirs()
    return sorted(PROCESSES_DIR.glob("*.rmp"))


def delete_process(path: Path) -> None:
    if path.exists():
        path.unlink()
        logger.info("Deleted process: %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Result I/O (ExampleSets, models, performance)
# ═══════════════════════════════════════════════════════════════════════════

def store_result(name: str, obj: Any) -> Path:
    """Pickle any Python object into RESULTS_DIR."""
    _ensure_dirs()
    safe = name.replace(" ", "_").replace("/", "_")
    path = RESULTS_DIR / f"{safe}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Stored result '%s' → %s", name, path)
    return path


def load_result(name: str) -> Any:
    """Load a pickled result by name."""
    safe = name.replace(" ", "_").replace("/", "_")
    path = RESULTS_DIR / f"{safe}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Result not found: {name}")
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def list_results() -> List[str]:
    """Return names of all stored results."""
    _ensure_dirs()
    return [p.stem for p in sorted(RESULTS_DIR.glob("*.pkl"))]


def delete_result(name: str) -> None:
    safe = name.replace(" ", "_").replace("/", "_")
    path = RESULTS_DIR / f"{safe}.pkl"
    if path.exists():
        path.unlink()
        logger.info("Deleted result '%s'", name)


# ═══════════════════════════════════════════════════════════════════════════
# Sample processes
# ═══════════════════════════════════════════════════════════════════════════

def list_sample_processes() -> List[Path]:
    """Return paths to bundled sample .rmp files."""
    if SAMPLE_DIR.exists():
        return sorted(SAMPLE_DIR.glob("*.rmp"))
    return []


def export_process(process: Process, dest: Path) -> Path:
    """Export a process to an arbitrary path."""
    return save_process(process, path=dest)
