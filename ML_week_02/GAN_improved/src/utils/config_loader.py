"""
YAML configuration loader with device auto-detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-detect device if set to "auto"
    if config.get("experiment", {}).get("device") == "auto":
        config["experiment"]["device"] = _detect_device()

    # Resolve relative paths to absolute (relative to project root)
    project_root = config_path.parent.parent
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str):
                config["paths"][key] = str(project_root / value)

    return config


def get_device(config: dict[str, Any] | None = None) -> torch.device:
    """Get the appropriate torch device.

    Priority: config setting > MPS > CUDA > CPU

    Args:
        config: Optional config dict; reads experiment.device if present.

    Returns:
        torch.device instance.
    """
    if config is not None:
        device_str = config.get("experiment", {}).get("device", "auto")
        if device_str != "auto":
            return torch.device(device_str)

    return torch.device(_detect_device())


def _detect_device() -> str:
    """Auto-detect the best available device.

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_project_root(config_path: str | Path) -> Path:
    """Infer project root from a config file path.

    Assumes configs live in <project_root>/config/.

    Args:
        config_path: Path to a config file.

    Returns:
        Project root as a Path object.
    """
    return Path(config_path).resolve().parent.parent


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two config dicts. override takes precedence.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
