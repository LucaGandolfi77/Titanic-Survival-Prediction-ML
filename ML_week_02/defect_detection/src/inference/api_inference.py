"""
api_inference.py – Async inference helpers for FastAPI integration.

Wraps DefectPredictor in an async-friendly way using thread-pool executors
for CPU-bound YOLO inference in an async web server.
"""
from __future__ import annotations

import asyncio
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from src.inference.predictor import DefectPredictor, PredictionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


async def predict_from_bytes(
    predictor: DefectPredictor,
    image_bytes: bytes,
    conf: float | None = None,
) -> PredictionResult:
    """Decode image bytes and run inference in a thread pool."""
    loop = asyncio.get_running_loop()

    def _run():
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image bytes")
        return predictor.predict_image(image, conf=conf)

    return await loop.run_in_executor(_executor, _run)


async def predict_from_upload(
    predictor: DefectPredictor,
    upload_file,
    conf: float | None = None,
) -> PredictionResult:
    """Read a FastAPI UploadFile and run inference."""
    contents = await upload_file.read()
    return await predict_from_bytes(predictor, contents, conf)


async def predict_video_from_bytes(
    predictor: DefectPredictor,
    video_bytes: bytes,
    stride: int = 1,
    max_frames: int = 300,
) -> List[PredictionResult]:
    """Process an uploaded video file frame-by-frame."""
    loop = asyncio.get_running_loop()

    def _run():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(tmp_path))
        results: List[PredictionResult] = []
        idx = 0
        while cap.isOpened() and len(results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                results.append(predictor.predict_image(frame))
            idx += 1
        cap.release()
        tmp_path.unlink(missing_ok=True)
        return results

    return await loop.run_in_executor(_executor, _run)


def encode_annotated_image(
    image: np.ndarray,
    detections: PredictionResult,
    fmt: str = ".jpg",
) -> bytes:
    """Draw detections and return encoded image bytes."""
    from src.utils.visualization import draw_detections as _draw

    if detections.detections:
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections.detections])
        classes = np.array([d.class_id for d in detections.detections])
        confs = np.array([d.confidence for d in detections.detections])
        image = _draw(image, boxes, classes, confs)

    _, buf = cv2.imencode(fmt, image)
    return buf.tobytes()
