"""Data loader for the Wisconsin Breast Cancer dataset.

Handles loading, splitting, scaling and optional CSV export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── project root (two levels up from this file) ──────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the YAML configuration file.

    Args:
        path: Filesystem path to config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class DataBundle:
    """Container for train/test splits and metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_names: list[str]
    scaler: StandardScaler


def load_data(
    config: Optional[dict] = None,
    scale: bool = True,
    export_csv: bool = True,
) -> DataBundle:
    """Load the breast-cancer dataset, split and optionally scale.

    Args:
        config: Configuration dict.  Loaded from disk when *None*.
        scale: Whether to apply StandardScaler.
        export_csv: Whether to save train/test CSVs under *outputs/*.

    Returns:
        A :class:`DataBundle` with all artefacts.
    """
    if config is None:
        config = load_config()

    random_state: int = config.get("random_state", 42)
    test_size: float = config.get("test_size", 0.2)

    # ── load raw data ────────────────────────────────────────────
    bunch = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = bunch.data  # type: ignore[assignment]
    y: pd.Series = bunch.target  # type: ignore[assignment]
    feature_names: list[str] = list(X.columns)
    target_names: list[str] = list(bunch.target_names)

    # ── stratified split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    if scale:
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index,
        )

    # ── optional CSV export ──────────────────────────────────────
    if export_csv:
        _export_splits(X_train, X_test, y_train, y_test, config)

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        scaler=scaler,
    )


def _export_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict,
) -> None:
    """Save train/test dataframes as CSV files.

    Args:
        X_train: Training features.
        X_test:  Testing features.
        y_train: Training labels.
        y_test:  Testing labels.
        config:  Configuration dictionary (for output paths).
    """
    out_dir = ROOT / config.get("paths", {}).get("outputs", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv(out_dir / "train.csv", index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(out_dir / "test.csv", index=False)


def get_feature_importances(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Compute univariate feature importances (absolute correlation with y).

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Sorted series of absolute correlations (descending).
    """
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    return correlations


# ── CLI convenience ──────────────────────────────────────────────
if __name__ == "__main__":
    bundle = load_data()
    print(f"Train set : {bundle.X_train.shape}")
    print(f"Test set  : {bundle.X_test.shape}")
    print(f"Features  : {len(bundle.feature_names)}")
    print(f"Classes   : {bundle.target_names}")
