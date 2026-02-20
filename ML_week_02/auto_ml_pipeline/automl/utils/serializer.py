"""Model serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .logger import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: Path) -> Path:
    """Persist a model to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Model saved → {path}  ({size_mb:.1f} MB)")
    return path


def load_model(path: Path) -> Any:
    """Load a joblib-serialized model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model
