"""
websocket.py – WebSocket endpoint for live-streaming detection.

Client sends image frames as binary, server responds with JSON detections.
"""
from __future__ import annotations

import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.inference.api_inference import predict_from_bytes
from src.inference.predictor import DefectPredictor
from src.utils.logging import get_logger

logger = get_logger(__name__)

ws_router = APIRouter()

_predictor: DefectPredictor | None = None


def set_ws_predictor(predictor: DefectPredictor) -> None:
    global _predictor
    _predictor = predictor


@ws_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Receive image frames as binary, return JSON detections.

    Protocol:
      Client → binary JPEG/PNG bytes
      Server → JSON {"detections": [...], "count": N, "inference_ms": X}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_bytes()

            if _predictor is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue

            result = await predict_from_bytes(_predictor, data)

            payload = {
                "detections": [d.to_dict() for d in result.detections],
                "count": result.count,
                "inference_ms": round(result.inference_ms, 2),
            }
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)
