"""
main.py – FastAPI application entry point.

Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_predictor
from src.api.websocket import ws_router, set_ws_predictor
from src.inference.predictor import DefectPredictor
from src.utils.config import load_api_config, load_inference_config, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    inf_cfg = load_inference_config()
    weights = project_root() / inf_cfg["model"]["weights"]
    device = inf_cfg["model"].get("device", "cpu")

    if not weights.exists():
        logger.warning(f"Weights not found at {weights} — using default yolov8n.pt")
        weights = "yolov8n.pt"

    predictor = DefectPredictor(
        weights=weights,
        device=device,
        conf=inf_cfg["detection"]["confidence_threshold"],
        iou=inf_cfg["detection"]["iou_threshold"],
        imgsz=inf_cfg["detection"]["image_size"],
    )
    set_predictor(predictor)
    set_ws_predictor(predictor)
    logger.success("Model loaded — API ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    api_cfg = load_api_config()["api"]

    application = FastAPI(
        title=api_cfg.get("title", "Defect Detection API"),
        version=api_cfg.get("version", "1.0.0"),
        description=api_cfg.get("description", ""),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router, tags=["Detection"])
    application.include_router(ws_router, tags=["WebSocket"])

    return application


app = create_app()
