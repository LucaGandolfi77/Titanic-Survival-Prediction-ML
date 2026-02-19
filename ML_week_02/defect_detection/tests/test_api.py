"""
test_api.py – Tests for FastAPI endpoints and Pydantic schemas.

Uses httpx AsyncClient so no real model is needed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.api.schemas import (
    BatchPredictionResponse,
    BoundingBox,
    DetectionSchema,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)


# ─────────────────────────────────────────────────────────────
# Schema tests (no server needed)
# ─────────────────────────────────────────────────────────────

class TestSchemas:
    def test_bounding_box(self):
        bb = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert bb.x1 == 1.0

    def test_detection_schema(self):
        d = DetectionSchema(
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.95,
            class_id=0,
            class_name="scratch",
        )
        assert d.confidence == 0.95

    def test_prediction_response(self):
        r = PredictionResponse(
            image_name="test.jpg",
            inference_ms=12.4,
            detections=[],
            count=0,
        )
        assert r.count == 0

    def test_batch_prediction_response(self):
        r = BatchPredictionResponse(results=[], total_images=0, avg_inference_ms=0.0)
        assert r.total_images == 0

    def test_health_response(self):
        r = HealthResponse(status="healthy", model_loaded=True, model_name="yolov8n", device="cpu", num_classes=5)
        assert r.model_loaded

    def test_metrics_response(self):
        r = MetricsResponse(total_predictions=10, avg_inference_ms=15.3, detections_per_class={"scratch": 5})
        assert r.detections_per_class["scratch"] == 5

    def test_error_response(self):
        r = ErrorResponse(detail="Not found")
        assert "Not found" in r.detail

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            DetectionSchema(
                bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
                confidence=1.5,  # invalid
                class_id=0,
                class_name="scratch",
            )


# ─────────────────────────────────────────────────────────────
# API route integration tests (mock model)
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_predictor():
    """Create a mock DefectPredictor that returns canned results."""
    from src.inference.predictor import Detection, PredictionResult

    pred = MagicMock()
    pred.device = "cpu"
    pred.model = MagicMock()
    pred.model.model_name = "yolov8n"
    pred.conf = 0.25
    pred.iou = 0.45
    pred.imgsz = 640

    result = PredictionResult(
        detections=[
            Detection(x1=10, y1=20, x2=100, y2=150, confidence=0.92, class_id=0, class_name="scratch"),
        ],
        image_path=None,
        inference_ms=15.4,
    )
    pred.predict_image.return_value = result
    return pred


@pytest.fixture()
def test_image_bytes():
    """A valid JPEG image in bytes."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture()
def app_client(mock_predictor):
    """Create a TestClient with a mocked predictor."""
    from fastapi.testclient import TestClient
    from src.api.routes import set_predictor, _stats, router
    from collections import Counter

    # Reset stats
    _stats["total"] = 0
    _stats["total_ms"] = 0.0
    _stats["per_class"] = Counter()

    set_predictor(mock_predictor)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestAPIRoutes:
    def test_health(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["num_classes"] == 5

    def test_metrics_empty(self, app_client):
        resp = app_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 0

    def test_predict_single(self, app_client, test_image_bytes, mock_predictor):
        # Patch predict_from_upload to simply call predictor.predict_image
        with patch("src.api.routes.predict_from_upload") as mock_upload:
            from src.inference.predictor import Detection, PredictionResult
            mock_upload.return_value = PredictionResult(
                detections=[Detection(x1=10, y1=20, x2=100, y2=150, confidence=0.92,
                                      class_id=0, class_name="scratch")],
                inference_ms=15.4,
            )
            resp = app_client.post(
                "/predict",
                files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["detections"][0]["class_name"] == "scratch"

    def test_predict_batch(self, app_client, test_image_bytes):
        with patch("src.api.routes.predict_from_upload") as mock_upload:
            from src.inference.predictor import PredictionResult
            mock_upload.return_value = PredictionResult(detections=[], inference_ms=5.0)
            resp = app_client.post(
                "/predict/batch",
                files=[
                    ("files", ("a.jpg", io.BytesIO(test_image_bytes), "image/jpeg")),
                    ("files", ("b.jpg", io.BytesIO(test_image_bytes), "image/jpeg")),
                ],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 2

    def test_metrics_after_predict(self, app_client, test_image_bytes):
        with patch("src.api.routes.predict_from_upload") as mock_upload:
            from src.inference.predictor import Detection, PredictionResult
            mock_upload.return_value = PredictionResult(
                detections=[Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.8,
                                      class_id=1, class_name="dent")],
                inference_ms=10.0,
            )
            app_client.post(
                "/predict",
                files={"file": ("t.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
            )
        resp = app_client.get("/metrics")
        data = resp.json()
        assert data["total_predictions"] >= 1
