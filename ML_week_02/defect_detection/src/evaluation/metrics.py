"""
metrics.py – Compute object detection evaluation metrics.

Works with YOLO results or raw prediction arrays.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import CLASS_NAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
    method: str = "interp11",
) -> float:
    """Average Precision from precision-recall arrays.

    Methods: 'interp11' (VOC 11-point) or 'all_points' (COCO).
    """
    if method == "interp11":
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t]
            ap += max(p) if len(p) > 0 else 0
        return ap / 11
    else:
        # All-points interpolation
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def match_predictions(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confs: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match predictions to ground-truth using IoU threshold.

    Returns:
        tp : (N,) boolean — True Positive mask
        fp : (N,) boolean — False Positive mask
        matched_gt : (N,) int — index of matched GT box (-1 if FP)
    """
    order = np.argsort(-pred_confs)
    pred_boxes = pred_boxes[order]
    pred_classes = pred_classes[order]
    pred_confs = pred_confs[order]

    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)
    matched_gt = -np.ones(len(pred_boxes), dtype=int)
    gt_matched = set()

    for i in range(len(pred_boxes)):
        best_iou = 0.0
        best_j = -1
        for j in range(len(gt_boxes)):
            if j in gt_matched:
                continue
            if gt_classes[j] != pred_classes[i]:
                continue
            iou = compute_iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            tp[i] = True
            matched_gt[i] = best_j
            gt_matched.add(best_j)
        else:
            fp[i] = True

    return tp, fp, matched_gt


def evaluate_predictions(
    gt_boxes_list: List[np.ndarray],
    gt_classes_list: List[np.ndarray],
    pred_boxes_list: List[np.ndarray],
    pred_classes_list: List[np.ndarray],
    pred_confs_list: List[np.ndarray],
    iou_thresholds: List[float] | None = None,
    class_names: Dict[int, str] | None = None,
) -> Dict:
    """Full evaluation across images and IoU thresholds.

    Returns dict with 'mAP50', 'mAP50_95', 'per_class', 'precision', 'recall'.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    names = class_names or CLASS_NAMES
    all_classes = sorted(set().union(*[set(g.tolist()) for g in gt_classes_list]))

    per_class: Dict[int, Dict] = {}
    aps_50: List[float] = []

    for cls_id in all_classes:
        cls_aps: List[float] = []
        for iou_t in iou_thresholds:
            all_tp, all_conf, n_gt = [], [], 0
            for gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf in zip(
                gt_boxes_list, gt_classes_list, pred_boxes_list, pred_classes_list, pred_confs_list
            ):
                mask_gt = gt_cls == cls_id
                mask_pred = pred_cls == cls_id
                n_gt += int(mask_gt.sum())

                if mask_pred.sum() == 0:
                    continue
                tp, fp, _ = match_predictions(
                    gt_boxes[mask_gt], gt_cls[mask_gt],
                    pred_boxes[mask_pred], pred_cls[mask_pred], pred_conf[mask_pred],
                    iou_threshold=iou_t,
                )
                all_tp.extend(tp.tolist())
                all_conf.extend(pred_conf[mask_pred].tolist())

            if n_gt == 0:
                cls_aps.append(0.0)
                continue

            order = np.argsort(-np.array(all_conf))
            tp_arr = np.array(all_tp)[order]
            cum_tp = np.cumsum(tp_arr)
            precision = cum_tp / np.arange(1, len(tp_arr) + 1)
            recall = cum_tp / n_gt
            ap = compute_ap(precision, recall, method="all_points")
            cls_aps.append(ap)

        per_class[cls_id] = {
            "name": names.get(cls_id, f"class_{cls_id}"),
            "AP50": round(cls_aps[0], 4) if cls_aps else 0.0,
            "AP50_95": round(float(np.mean(cls_aps)), 4),
        }
        aps_50.append(cls_aps[0] if cls_aps else 0.0)

    return {
        "mAP50": round(float(np.mean(aps_50)), 4) if aps_50 else 0.0,
        "mAP50_95": round(float(np.mean([v["AP50_95"] for v in per_class.values()])), 4),
        "per_class": per_class,
    }
