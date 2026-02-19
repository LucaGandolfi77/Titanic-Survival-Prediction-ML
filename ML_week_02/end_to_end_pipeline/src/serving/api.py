"""
api.py — FastAPI Application for Titanic Survival Prediction
=============================================================
Production-ready REST API with:
  • Single & batch prediction endpoints
  • Health check / readiness probe
  • Request logging middleware
  • Prometheus-compatible metrics stub
  • OpenAPI documentation auto-generated
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.predictor import TitanicPredictor
from src.serving.schemas import (
    BatchInput,
    BatchOutput,
    HealthResponse,
    PassengerInput,
    PredictionOutput,
)
from src.utils.config_loader import PROJECT_ROOT, load_config, setup_logging

logger = logging.getLogger("titanic_mlops.api")

# ── App factory ──────────────────────────────────────────────

def create_app() -> FastAPI:
    """Application factory — builds and configures the FastAPI instance."""

    setup_logging()
    cfg = load_config()

    app = FastAPI(
        title="Titanic Survival Prediction API",
        description=(
            "Production-grade ML inference service.  "
            "Accepts passenger features and returns survival predictions."
        ),
        version=cfg["project"]["version"],
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── State: predictor instance ────────────────────────────
    models_dir: Path = cfg["paths"]["models_dir"]
    processed_dir: Path = cfg["paths"]["processed_dir"]

    # Try loading the best available model
    model_path = models_dir / "xgboost.joblib"
    if not model_path.exists():
        # Fall back to any available joblib model
        candidates = sorted(models_dir.glob("*.joblib"))
        model_path = candidates[0] if candidates else None

    preprocessor_path = processed_dir / "preprocessor.pkl"

    predictor = TitanicPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path if preprocessor_path.exists() else None,
    )
    app.state.predictor = predictor
    app.state.config = cfg

    # ── Request logging middleware ───────────────────────────

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # ── Routes ───────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["monitoring"])
    async def health_check():
        """Kubernetes-style liveness / readiness probe."""
        return HealthResponse(
            status="healthy" if predictor.is_ready else "degraded",
            model_loaded=predictor.is_ready,
            version=cfg["project"]["version"],
        )

    @app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
    async def predict_single(passenger: PassengerInput):
        """
        Predict survival for a **single** passenger.

        Provide passenger features in the request body.
        """
        if not predictor.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            result = predictor.predict_single(passenger.model_dump())
            return PredictionOutput(**result)
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/predict/batch", response_model=BatchOutput, tags=["prediction"])
    async def predict_batch(batch: BatchInput):
        """
        Predict survival for a **batch** of passengers.

        Send an array of passenger records for bulk inference.
        """
        if not predictor.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            passengers = [p.model_dump() for p in batch.passengers]
            results = predictor.predict(passengers)
            return BatchOutput(
                predictions=[PredictionOutput(**r) for r in results]
            )
        except Exception as exc:
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/model/info", tags=["monitoring"])
    async def model_info():
        """Return metadata about the currently loaded model."""
        return {
            "model_loaded": predictor.is_ready,
            "model_type": type(predictor.model).__name__ if predictor.model else None,
            "project": cfg["project"],
        }

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
