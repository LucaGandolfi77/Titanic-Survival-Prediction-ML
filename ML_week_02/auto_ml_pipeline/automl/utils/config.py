"""Configuration management — load YAML configs as nested attribute dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


class AttrDict(dict):
    """Dict that allows attribute-style access (config.eda.outliers.method)."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return AttrDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config key not found: '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]


def _to_attrdict(obj: Any) -> Any:
    """Recursively convert nested dicts to AttrDict."""
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(i) for i in obj]
    return obj


def load_config(path: Optional[Path] = None) -> AttrDict:
    """Load YAML config, falling back to default.yaml bundled with the package.

    Args:
        path: Explicit YAML path.  When *None*, looks for
              ``configs/default.yaml`` relative to the project root.

    Returns:
        AttrDict with nested attribute access.
    """
    if path is None:
        path = project_root() / "configs" / "default.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    return _to_attrdict(raw or {})


def project_root() -> Path:
    """Return the auto_ml_pipeline project root."""
    # config.py lives at automl/utils/config.py → 3 levels up
    return Path(__file__).resolve().parent.parent.parent
