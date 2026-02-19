"""
predictor.py – Single-image / batch inference wrapper.

Loads a YOLOv8 model (PT or ONNX) and provides a clean prediction API
returning structured results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.utils.config import CLASS_NAMES, load_inference_config, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """One detected bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> Dict:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class PredictionResult:
    """Collection of detections for a single image."""

    detections: List[Detection] = field(default_factory=list)
    image_path: Optional[str] = None
    inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> "PredictionResult":
        return PredictionResult(
            detections=[d for d in self.detections if d.confidence >= threshold],
            image_path=self.image_path,
            inference_ms=self.inference_ms,
        )

    def to_dict(self) -> Dict:
        return {
            "image": self.image_path,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
        }


class DefectPredictor:
    """High-level YOLO predictor for defect detection."""

    def __init__(
        self,
        weights: Path | str | None = None,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        if weights is None:
            cfg = load_inference_config()
            weights = project_root() / cfg["model"]["weights"]
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        logger.info(f"DefectPredictor ready — weights={weights}, device={device}")

    def predict_image(
        self,
        image: np.ndarray | str | Path,
        conf: float | None = None,
        iou: float | None = None,
    ) -> PredictionResult:
        """Run detection on a single image (path or numpy BGR array)."""
        import time

        t0 = time.perf_counter()
        results = self.model.predict(
            source=str(image) if isinstance(image, Path) else image,
            device=self.device,
            conf=conf or self.conf,
            iou=iou or self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        detections = self._parse_results(results[0])
        return PredictionResult(
            detections=detections,
            image_path=str(image) if isinstance(image, (str, Path)) else None,
            inference_ms=elapsed,
        )

    def predict_batch(
        self,
        images: Sequence[np.ndarray | str | Path],
        conf: float | None = None,
    ) -> List[PredictionResult]:
        """Run detection on a batch of images."""
        import time

        sources = [str(img) if isinstance(img, Path) else img for img in images]
        t0 = time.perf_counter()
        results = self.model.predict(
            source=sources,
            device=self.device,
            conf=conf or self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        out: List[PredictionResult] = []
        for i, r in enumerate(results):
            dets = self._parse_results(r)
            out.append(PredictionResult(
                detections=dets,
                image_path=str(images[i]) if isinstance(images[i], (str, Path)) else None,
                inference_ms=elapsed / len(results),
            ))
        return out

    def _parse_results(self, result) -> List[Detection]:
        """Convert Ultralytics result to a list of Detection."""
        dets: List[Detection] = []
        if result.boxes is None:
            return dets
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2,
                                  confidence=conf, class_id=cls_id, class_name=cls_name))
        return dets
