"""
config_loader.py — YAML Configuration Management
=================================================
Single source of truth for all project settings.
Loads config/config.yaml and exposes it as a typed dictionary
with resolved pathlib.Path objects.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root: two levels up from this file (src/utils -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load the master configuration.

    Parameters
    ----------
    config_path : Path, optional
        Override the default config location.

    Returns
    -------
    dict
        Fully-resolved configuration dictionary with Path objects.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"

    cfg = load_yaml(config_path)

    # Resolve relative paths to absolute paths anchored at PROJECT_ROOT
    for key in ("data_dir", "processed_dir", "models_dir", "logs_dir"):
        cfg["paths"][key] = PROJECT_ROOT / cfg["paths"][key]

    # Ensure output directories exist
    for key in ("processed_dir", "models_dir", "logs_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    return cfg


def setup_logging(logging_config_path: Path | None = None) -> None:
    """
    Configure the Python logging subsystem from a YAML file.

    Parameters
    ----------
    logging_config_path : Path, optional
        Override the default logging config location.
    """
    if logging_config_path is None:
        logging_config_path = PROJECT_ROOT / "config" / "logging_config.yaml"

    log_cfg = load_yaml(logging_config_path)

    # Resolve log file path relative to project root
    file_handler = log_cfg.get("handlers", {}).get("file", {})
    if "filename" in file_handler:
        log_file = PROJECT_ROOT / file_handler["filename"]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler["filename"] = str(log_file)

    logging.config.dictConfig(log_cfg)
