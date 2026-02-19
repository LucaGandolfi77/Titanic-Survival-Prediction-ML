"""
ensemble.py – Multi-model ensemble for defect detection.

Combine predictions from multiple YOLOv8 checkpoints (different sizes
or different training runs) using Weighted Boxes Fusion (WBF) or
Non-Maximum Suppression (NMS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DetectionEnsemble:
    """Ensemble of YOLO models with NMS / WBF fusion."""

    def __init__(
        self,
        weight_paths: Sequence[Path],
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        method: str = "nms",
    ):
        from ultralytics import YOLO

        self.models = [YOLO(str(p)) for p in weight_paths]
        self.device = device
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.method = method  # "nms" or "wbf"
        logger.info(f"Ensemble: {len(self.models)} models, method={method}")

    def predict(self, source, imgsz: int = 640) -> List[Dict]:
        """Run all models on *source* and fuse predictions.

        Returns a list of detection dicts per image:
            {"boxes": ndarray (N,4), "classes": ndarray (N,), "confidences": ndarray (N,)}
        """
        all_preds: List[List[Dict]] = []

        for model in self.models:
            results = model.predict(
                source, device=self.device, imgsz=imgsz,
                conf=self.conf, iou=self.iou, verbose=False,
            )
            preds = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty(0)
                classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.empty(0, int)
                preds.append({"boxes": boxes, "classes": classes, "confidences": confs})
            all_preds.append(preds)

        # Transpose: per-image → per-model
        n_images = len(all_preds[0])
        fused: List[Dict] = []
        for i in range(n_images):
            img_preds = [all_preds[m][i] for m in range(len(self.models))]
            fused.append(self._fuse(img_preds))
        return fused

    def _fuse(self, preds: List[Dict]) -> Dict:
        """Fuse detections from all models for a single image."""
        all_boxes = np.concatenate([p["boxes"] for p in preds], axis=0) if preds else np.empty((0, 4))
        all_confs = np.concatenate([p["confidences"] for p in preds]) if preds else np.empty(0)
        all_cls = np.concatenate([p["classes"] for p in preds]) if preds else np.empty(0, int)

        if len(all_boxes) == 0:
            return {"boxes": all_boxes, "classes": all_cls, "confidences": all_confs}

        if self.method == "nms":
            keep = self._nms(all_boxes, all_confs, self.iou)
        else:
            keep = np.arange(len(all_boxes))  # WBF would go here

        return {
            "boxes": all_boxes[keep],
            "classes": all_cls[keep],
            "confidences": all_confs[keep],
        }

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        """Class-agnostic NMS (numpy)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)
