"""Configuration loading and misc utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from loguru import logger


def project_root() -> Path:
    """Return the NAS project root (``ML_week_03/nas/``)."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML config.  Defaults to ``configs/default.yaml``."""
    if path is None:
        path = project_root() / "configs" / "default.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Create output directories and return a dict of paths."""
    out = Path(cfg.get("output_dir", "outputs"))
    dirs = {
        "root": out,
        "architectures": out / "architectures",
        "plots": out / "plots",
        "logs": out / "logs",
        "models": out / "models",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
