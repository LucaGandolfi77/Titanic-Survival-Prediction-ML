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
            return {
                "boxes": all_boxes[keep],
                "classes": all_cls[keep],
                "confidences": all_confs[keep],
            }
        else:
            return self._wbf(preds, self.iou)


    def _wbf(
        self,
        preds: List[Dict],
        iou_thresh: float,
        skip_box_thr: float = 0.01,
    ) -> Dict:
        """Weighted Boxes Fusion — fuse overlapping boxes from multiple models.

        For each cluster of IoU-matching boxes across models, the fused box
        is the confidence-weighted average of coordinates, and the fused
        confidence is the (clipped) sum of individual scores.
        """
        n_models = len(preds)
        all_boxes = np.concatenate([p["boxes"] for p in preds], axis=0)
        all_confs = np.concatenate([p["confidences"] for p in preds])
        all_cls = np.concatenate([p["classes"] for p in preds])

        # Filter low-confidence boxes
        mask = all_confs >= skip_box_thr
        all_boxes, all_confs, all_cls = all_boxes[mask], all_confs[mask], all_cls[mask]

        if len(all_boxes) == 0:
            return {"boxes": np.empty((0, 4)), "classes": np.empty(0, int), "confidences": np.empty(0)}

        fused_boxes, fused_confs, fused_cls = [], [], []

        for cls_id in np.unique(all_cls):
            cls_mask = all_cls == cls_id
            c_boxes = all_boxes[cls_mask]
            c_confs = all_confs[cls_mask]

            # Sort by confidence descending
            order = c_confs.argsort()[::-1]
            c_boxes, c_confs = c_boxes[order], c_confs[order]

            # Greedy cluster formation
            clusters: List[Tuple[List[np.ndarray], List[float]]] = []
            used = np.zeros(len(c_boxes), dtype=bool)

            for i in range(len(c_boxes)):
                if used[i]:
                    continue
                cluster_boxes = [c_boxes[i]]
                cluster_confs = [c_confs[i]]
                used[i] = True

                for j in range(i + 1, len(c_boxes)):
                    if used[j]:
                        continue
                    iou = self._compute_iou(c_boxes[i], c_boxes[j])
                    if iou > iou_thresh:
                        cluster_boxes.append(c_boxes[j])
                        cluster_confs.append(c_confs[j])
                        used[j] = True
                clusters.append((cluster_boxes, cluster_confs))

            for cluster_boxes, cluster_confs in clusters:
                weights = np.array(cluster_confs)
                weighted_box = np.average(np.array(cluster_boxes), axis=0, weights=weights)
                # Fused confidence = sum of scores / number of models (clipped to 1)
                fused_conf = min(float(weights.sum()) / n_models, 1.0)
                fused_boxes.append(weighted_box)
                fused_confs.append(fused_conf)
                fused_cls.append(cls_id)

        return {
            "boxes": np.array(fused_boxes) if fused_boxes else np.empty((0, 4)),
            "classes": np.array(fused_cls, dtype=int) if fused_cls else np.empty(0, int),
            "confidences": np.array(fused_confs) if fused_confs else np.empty(0),
        }

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """IoU between two xyxy boxes."""
        xx1 = max(box_a[0], box_b[0])
        yy1 = max(box_a[1], box_b[1])
        xx2 = min(box_a[2], box_b[2])
        yy2 = min(box_a[3], box_b[3])
        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

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
