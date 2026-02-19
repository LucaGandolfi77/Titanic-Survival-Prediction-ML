"""
schemas.py – Pydantic models for the FastAPI defect detection API.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSchema(BaseModel):
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    class_id: int
    class_name: str


class PredictionResponse(BaseModel):
    image_name: Optional[str] = None
    inference_ms: float
    detections: List[DetectionSchema]
    count: int


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_images: int
    avg_inference_ms: float


class VideoFrameResponse(BaseModel):
    frame_index: int
    detections: List[DetectionSchema]
    count: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_name: str
    device: str
    num_classes: int


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_inference_ms: float
    detections_per_class: Dict[str, int]


class ErrorResponse(BaseModel):
    detail: str
