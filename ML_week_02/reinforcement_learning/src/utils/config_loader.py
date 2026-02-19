"""YAML configuration loader with device auto-detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and inject computed fields.

    Adds:
    * ``experiment.device`` → resolved device string.
    * ``paths.*`` → resolved to absolute ``pathlib.Path`` objects.
    """
    path = Path(path)
    with path.open("r") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    # Resolve device
    exp = config.setdefault("experiment", {})
    requested = exp.get("device", "auto")
    exp["device"] = _resolve_device(requested)

    # Resolve paths relative to config file location
    project_root = path.parent.parent  # config/ → project root
    paths = config.get("paths", {})
    for key, val in paths.items():
        if isinstance(val, str):
            resolved = project_root / val
            resolved.mkdir(parents=True, exist_ok=True)
            paths[key] = str(resolved)
    config["paths"] = paths

    return config


def get_device(config: dict[str, Any]) -> torch.device:
    """Return ``torch.device`` from a loaded config."""
    dev_str = config.get("experiment", {}).get("device", "cpu")
    return torch.device(dev_str)


def _resolve_device(requested: str) -> str:
    """Auto-detect the best available device.

    Priority: MPS (Apple Silicon) → CUDA → CPU.
    """
    if requested != "auto":
        return requested

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def merge_configs(base: dict, overrides: dict) -> dict:
    """Deep-merge *overrides* into *base* (non-destructive)."""
    merged = base.copy()
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged
