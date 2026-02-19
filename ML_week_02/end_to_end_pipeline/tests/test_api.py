"""
test_api.py — Unit Tests for the FastAPI Application
=====================================================
Uses the HTTPX test client provided by FastAPI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_predictor():
    """A mocked TitanicPredictor."""
    pred = MagicMock()
    pred.is_ready = True
    pred.predict_single.return_value = {"survived": 1, "probability": 0.85}
    pred.predict.return_value = [
        {"survived": 1, "probability": 0.85},
        {"survived": 0, "probability": 0.15},
    ]
    pred.model = MagicMock()
    pred.model.__class__.__name__ = "XGBClassifier"
    return pred


@pytest.fixture
def client(mock_predictor) -> TestClient:
    """
    Create a TestClient with a mocked predictor injected.
    We patch the TitanicPredictor constructor so no real files are needed.
    """
    with patch("src.serving.api.TitanicPredictor", return_value=mock_predictor):
        from src.serving.api import create_app
        app = create_app()
        app.state.predictor = mock_predictor
        return TestClient(app)


@pytest.fixture
def sample_payload() -> dict:
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


# ── Health check ─────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "model_loaded" in data
        assert "version" in data


# ── Single prediction ────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "survived" in data
        assert "probability" in data

    def test_predict_validation_error(self, client):
        """Missing required fields should return 422."""
        resp = client.post("/predict", json={"Pclass": 3})
        assert resp.status_code == 422

    def test_predict_pclass_bounds(self, client, sample_payload):
        sample_payload["Pclass"] = 5  # invalid
        resp = client.post("/predict", json=sample_payload)
        assert resp.status_code == 422


# ── Batch prediction ────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client, sample_payload):
        resp = client.post(
            "/predict/batch",
            json={"passengers": [sample_payload, sample_payload]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2

    def test_batch_empty_list(self, client, mock_predictor):
        mock_predictor.predict.return_value = []
        resp = client.post("/predict/batch", json={"passengers": []})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 0


# ── Model info ───────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_loaded" in data
        assert "project" in data
