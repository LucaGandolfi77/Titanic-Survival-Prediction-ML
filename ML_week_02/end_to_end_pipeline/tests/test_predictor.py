"""
test_predictor.py — Unit Tests for the Inference Wrapper
=========================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.predictor import TitanicPredictor


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """A mock sklearn model with predict and predict_proba."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def mock_preprocessor():
    """A mock ColumnTransformer that returns a fixed-size array."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    return preprocessor


@pytest.fixture
def predictor(mock_model, mock_preprocessor) -> TitanicPredictor:
    return TitanicPredictor(model=mock_model, preprocessor=mock_preprocessor)


@pytest.fixture
def sample_passenger() -> dict:
    return {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Name": "Braund, Mr. Owen Harris",
    }


# ── Tests ────────────────────────────────────────────────────

class TestTitanicPredictor:
    def test_is_ready(self, predictor: TitanicPredictor):
        assert predictor.is_ready is True

    def test_not_ready_without_model(self, mock_preprocessor):
        pred = TitanicPredictor(preprocessor=mock_preprocessor)
        assert pred.is_ready is False

    def test_not_ready_without_preprocessor(self, mock_model):
        pred = TitanicPredictor(model=mock_model)
        assert pred.is_ready is False

    def test_predict_single(self, predictor, sample_passenger):
        result = predictor.predict_single(sample_passenger)
        assert "survived" in result
        assert "probability" in result
        assert result["survived"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_batch(self, predictor, sample_passenger, mock_model):
        # Adjust mock for batch of 2
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

        results = predictor.predict([sample_passenger, sample_passenger])
        assert len(results) == 2
        assert results[0]["survived"] == 1
        assert results[1]["survived"] == 0

    def test_predict_raises_when_not_ready(self, sample_passenger):
        pred = TitanicPredictor()
        with pytest.raises(RuntimeError, match="not initialised"):
            pred.predict_single(sample_passenger)

    def test_probability_rounded(self, predictor, sample_passenger):
        result = predictor.predict_single(sample_passenger)
        # Check that probability is rounded to 4 decimal places
        prob_str = str(result["probability"])
        if "." in prob_str:
            decimal_places = len(prob_str.split(".")[1])
            assert decimal_places <= 4
