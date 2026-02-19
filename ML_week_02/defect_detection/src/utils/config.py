"""
config.py – Centralised configuration management.

Loads YAML configs with pathlib, merges defaults, and exposes typed helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # defect_detection/


def project_root() -> Path:
    """Return the absolute project root path."""
    return _PROJECT_ROOT


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and return a YAML file as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def load_training_config(name: str = "base") -> Dict[str, Any]:
    """Load a training configuration by name."""
    cfg_path = _PROJECT_ROOT / "configs" / "training" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")
    return load_yaml(cfg_path)


def load_inference_config() -> Dict[str, Any]:
    """Load inference / deployment config."""
    cfg_path = _PROJECT_ROOT / "configs" / "deployment" / "inference_config.yaml"
    return load_yaml(cfg_path)


def load_api_config() -> Dict[str, Any]:
    """Load FastAPI config."""
    cfg_path = _PROJECT_ROOT / "configs" / "deployment" / "api_config.yaml"
    return load_yaml(cfg_path)


def dataset_yaml_path() -> Path:
    """Return the default dataset.yaml path."""
    return _PROJECT_ROOT / "data" / "dataset.yaml"


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dicts (later values win)."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


# ── Class-name helpers ───────────────────────────────────────
CLASS_NAMES: Dict[int, str] = {
    0: "scratch",
    1: "dent",
    2: "discoloration",
    3: "crack",
    4: "missing_component",
}

NUM_CLASSES = len(CLASS_NAMES)


def class_name(idx: int) -> str:
    return CLASS_NAMES.get(idx, f"class_{idx}")


def class_colours() -> Dict[str, tuple]:
    """BGR colour palette used for drawing bounding boxes."""
    return {
        "scratch": (53, 107, 255),
        "dent": (244, 133, 66),
        "discoloration": (83, 168, 52),
        "crack": (53, 67, 234),
        "missing_component": (176, 39, 156),
    }
