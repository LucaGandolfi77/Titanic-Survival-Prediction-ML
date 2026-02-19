"""
loader.py – Load pre-trained models from disk.

Supports scikit-learn, XGBoost, LightGBM and CatBoost models
serialised with joblib or pickle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.base import BaseEstimator

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _models_dir(override: Optional[Path] = None) -> Path:
    return override if override is not None else MODELS_DIR


def list_available_models(models_dir: Optional[Path] = None) -> list[str]:
    """Return names (stems) of all .pkl files in the models/ directory."""
    d = _models_dir(models_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.pkl"))


def load_model(name: str, models_dir: Optional[Path] = None) -> BaseEstimator:
    """Load a model by stem name (e.g. 'credit_risk_model')."""
    d = _models_dir(models_dir)
    path = d / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_metadata(name: str, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the JSON metadata file for a specific model."""
    d = _models_dir(models_dir)
    meta_path = d / f"{name}_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_model_metadata(name: str, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Alias for load_metadata."""
    return load_metadata(name, models_dir)


def save_model(estimator: BaseEstimator, name: str, models_dir: Optional[Path] = None) -> Path:
    """Persist a trained model as .pkl."""
    d = _models_dir(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pkl"
    joblib.dump(estimator, path)
    return path


def save_metadata(name: str, meta: Dict[str, Any], models_dir: Optional[Path] = None) -> Path:
    """Persist metadata JSON for a specific model."""
    d = _models_dir(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / f"{name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
