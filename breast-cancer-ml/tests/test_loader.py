"""Tests for data.loader module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import DataBundle, load_config, load_data, get_feature_importances


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def config() -> dict:
    """Load shared config once per module."""
    return load_config()


@pytest.fixture(scope="module")
def bundle(config: dict) -> DataBundle:
    """Load dataset once per module (no CSV export to avoid side-effects)."""
    return load_data(config, scale=True, export_csv=False)


# ── Tests ────────────────────────────────────────────────────────
class TestLoadConfig:
    """Tests for load_config()."""

    def test_returns_dict(self, config: dict) -> None:
        assert isinstance(config, dict)

    def test_has_random_state(self, config: dict) -> None:
        assert "random_state" in config
        assert config["random_state"] == 42

    def test_has_models_section(self, config: dict) -> None:
        assert "models" in config
        assert "logistic_regression" in config["models"]


class TestLoadData:
    """Tests for load_data()."""

    def test_returns_databundle(self, bundle: DataBundle) -> None:
        assert isinstance(bundle, DataBundle)

    def test_train_test_shapes(self, bundle: DataBundle) -> None:
        total = len(bundle.X_train) + len(bundle.X_test)
        assert total == 569
        assert bundle.X_train.shape[1] == 30
        assert bundle.X_test.shape[1] == 30

    def test_stratified_split(self, bundle: DataBundle) -> None:
        """Class proportions should be similar in train and test."""
        train_ratio = bundle.y_train.mean()
        test_ratio = bundle.y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05

    def test_scaling_zero_mean(self, bundle: DataBundle) -> None:
        """Scaled training features should have near-zero mean."""
        means = bundle.X_train.mean()
        assert np.allclose(means, 0, atol=1e-10)

    def test_feature_names(self, bundle: DataBundle) -> None:
        assert len(bundle.feature_names) == 30
        assert "mean radius" in bundle.feature_names

    def test_target_names(self, bundle: DataBundle) -> None:
        assert "malignant" in bundle.target_names
        assert "benign" in bundle.target_names

    def test_scaler_present(self, bundle: DataBundle) -> None:
        assert bundle.scaler is not None
        assert hasattr(bundle.scaler, "mean_")


class TestFeatureImportances:
    """Tests for get_feature_importances()."""

    def test_returns_sorted_series(self, bundle: DataBundle) -> None:
        imp = get_feature_importances(bundle.X_train, bundle.y_train)
        assert isinstance(imp, pd.Series)
        # Should be sorted descending
        assert imp.iloc[0] >= imp.iloc[-1]

    def test_all_positive(self, bundle: DataBundle) -> None:
        imp = get_feature_importances(bundle.X_train, bundle.y_train)
        assert (imp >= 0).all()

    def test_length_matches_features(self, bundle: DataBundle) -> None:
        imp = get_feature_importances(bundle.X_train, bundle.y_train)
        assert len(imp) == 30
