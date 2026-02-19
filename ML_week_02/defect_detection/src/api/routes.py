"""
routes.py – FastAPI endpoint definitions.

Endpoints:
  POST  /predict          – single image
  POST  /predict/batch    – multiple images
  POST  /predict/video    – video file
  GET   /health           – healthcheck
  GET   /metrics          – prediction metrics
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Query, UploadFile, HTTPException
from fastapi.responses import Response

from src.api.schemas import (
    BatchPredictionResponse,
    BoundingBox,
    DetectionSchema,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)
from src.inference.api_inference import predict_from_upload, predict_video_from_bytes
from src.inference.predictor import DefectPredictor, PredictionResult
from src.utils.config import CLASS_NAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Shared state — set by main.py at startup
_predictor: DefectPredictor | None = None
_stats: dict = {"total": 0, "total_ms": 0.0, "per_class": Counter()}


def set_predictor(predictor: DefectPredictor) -> None:
    global _predictor
    _predictor = predictor


def _get_predictor() -> DefectPredictor:
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    return _predictor


def _result_to_schema(result: PredictionResult, name: str | None = None) -> PredictionResponse:
    dets = []
    for d in result.detections:
        dets.append(DetectionSchema(
            bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
            confidence=round(d.confidence, 4),
            class_id=d.class_id,
            class_name=d.class_name,
        ))
    return PredictionResponse(
        image_name=name,
        inference_ms=round(result.inference_ms, 2),
        detections=dets,
        count=result.count,
    )


# ── POST /predict ────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.05, le=0.95),
):
    """Detect defects in a single uploaded image."""
    predictor = _get_predictor()
    result = await predict_from_upload(predictor, file, conf=confidence)

    _stats["total"] += 1
    _stats["total_ms"] += result.inference_ms
    for d in result.detections:
        _stats["per_class"][d.class_name] += 1

    return _result_to_schema(result, name=file.filename)


# ── POST /predict/batch ──────────────────────────────────────

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Query(0.25, ge=0.05, le=0.95),
):
    """Detect defects in multiple uploaded images."""
    predictor = _get_predictor()
    responses: List[PredictionResponse] = []
    total_ms = 0.0
    for f in files:
        result = await predict_from_upload(predictor, f, conf=confidence)
        responses.append(_result_to_schema(result, name=f.filename))
        total_ms += result.inference_ms
        _stats["total"] += 1
        _stats["total_ms"] += result.inference_ms
        for d in result.detections:
            _stats["per_class"][d.class_name] += 1

    return BatchPredictionResponse(
        results=responses,
        total_images=len(responses),
        avg_inference_ms=round(total_ms / max(len(responses), 1), 2),
    )


# ── POST /predict/video ──────────────────────────────────────

@router.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    stride: int = Query(1, ge=1, le=30),
    max_frames: int = Query(300, ge=1, le=3000),
    confidence: float = Query(0.25, ge=0.05, le=0.95),
):
    """Process a video file frame-by-frame."""
    predictor = _get_predictor()
    contents = await file.read()
    results = await predict_video_from_bytes(predictor, contents, stride=stride, max_frames=max_frames)
    frames = []
    for i, r in enumerate(results):
        frames.append({
            "frame": i,
            "detections": [d.to_dict() for d in r.detections],
            "count": r.count,
            "inference_ms": round(r.inference_ms, 2),
        })
    return {"filename": file.filename, "frames_processed": len(results), "frames": frames}


# ── GET /health ──────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    predictor = _get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=str(predictor.model.model_name) if hasattr(predictor.model, "model_name") else "yolov8",
        device=predictor.device,
        num_classes=len(CLASS_NAMES),
    )


# ── GET /metrics ─────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(
        total_predictions=_stats["total"],
        avg_inference_ms=round(_stats["total_ms"] / max(_stats["total"], 1), 2),
        detections_per_class=dict(_stats["per_class"]),
    )
