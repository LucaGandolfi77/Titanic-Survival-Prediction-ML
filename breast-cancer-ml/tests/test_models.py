"""Tests for model wrappers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Type

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import DataBundle, load_config, load_data
from models.base_model import BaseModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.xgboost_model import XGBoostModel


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bundle() -> DataBundle:
    config = load_config()
    return load_data(config, scale=True, export_csv=False)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


MODEL_CLASSES: list[Type[BaseModel]] = [
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
]


# ── Tests ────────────────────────────────────────────────────────
class TestBaseModelInterface:
    """Verify the abstract interface is properly implemented."""

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_is_subclass_of_base(self, cls: Type[BaseModel]) -> None:
        assert issubclass(cls, BaseModel)

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_has_required_methods(self, cls: Type[BaseModel]) -> None:
        instance = cls()
        assert hasattr(instance, "fit")
        assert hasattr(instance, "predict")
        assert hasattr(instance, "predict_proba")
        assert hasattr(instance, "evaluate")

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_has_name(self, cls: Type[BaseModel]) -> None:
        instance = cls()
        assert isinstance(instance.name, str)
        assert len(instance.name) > 0


class TestModelTraining:
    """Fit each model on a small dataset and verify predictions."""

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_fit_and_predict(
        self, cls: Type[BaseModel], bundle: DataBundle
    ) -> None:
        model = cls({"random_state": 42})
        model.fit(bundle.X_train, bundle.y_train)
        preds = model.predict(bundle.X_test)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(bundle.X_test)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_predict_proba_shape(
        self, cls: Type[BaseModel], bundle: DataBundle
    ) -> None:
        model = cls({"random_state": 42})
        model.fit(bundle.X_train, bundle.y_train)
        proba = model.predict_proba(bundle.X_test)

        assert proba.shape == (len(bundle.X_test), 2)
        # Probabilities must sum to ~1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_evaluate_returns_metrics(
        self, cls: Type[BaseModel], bundle: DataBundle
    ) -> None:
        model = cls({"random_state": 42})
        model.fit(bundle.X_train, bundle.y_train)
        metrics = model.evaluate(bundle.X_test, bundle.y_test)

        assert isinstance(metrics, dict)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    @pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.name)
    def test_accuracy_above_threshold(
        self, cls: Type[BaseModel], bundle: DataBundle
    ) -> None:
        """All models should beat random (>0.70) on this easy dataset."""
        model = cls({"random_state": 42})
        model.fit(bundle.X_train, bundle.y_train)
        metrics = model.evaluate(bundle.X_test, bundle.y_test)
        assert metrics["accuracy"] > 0.70


class TestModelSaveLoad:
    """Verify model serialisation round-trip."""

    def test_save_and_load(self, bundle: DataBundle, tmp_path: Path) -> None:
        model = LogisticRegressionModel({"random_state": 42})
        model.fit(bundle.X_train, bundle.y_train)

        pkl_path = tmp_path / "test_model.pkl"
        model.save(pkl_path)
        assert pkl_path.exists()

        loaded_estimator = BaseModel.load(pkl_path)
        preds_original = model.predict(bundle.X_test)
        preds_loaded = loaded_estimator.predict(bundle.X_test)
        assert np.array_equal(preds_original, preds_loaded)
