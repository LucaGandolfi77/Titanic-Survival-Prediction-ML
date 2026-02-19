"""
confusion_matrix.py – Per-class confusion matrix for object detection.

Adapted for IoU-matched detection results (not pixel classification).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

from src.evaluation.metrics import match_predictions
from src.utils.config import CLASS_NAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_confusion_matrix(
    gt_boxes_list: List[np.ndarray],
    gt_classes_list: List[np.ndarray],
    pred_boxes_list: List[np.ndarray],
    pred_classes_list: List[np.ndarray],
    pred_confs_list: List[np.ndarray],
    num_classes: int = 5,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Build an (num_classes+1 × num_classes+1) confusion matrix.

    Extra row/column for "background" (FP with no GT / unmatched GT = FN).
    """
    n = num_classes + 1  # last index = background
    cm = np.zeros((n, n), dtype=int)

    for gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf in zip(
        gt_boxes_list, gt_classes_list, pred_boxes_list, pred_classes_list, pred_confs_list
    ):
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            for c in gt_cls:
                cm[int(c), n - 1] += 1  # FN → GT→bg
            continue

        if len(gt_boxes) == 0:
            for c in pred_cls:
                cm[n - 1, int(c)] += 1  # FP → bg→pred
            continue

        tp, fp, matched_gt = match_predictions(
            gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, iou_threshold
        )
        order = np.argsort(-pred_conf)
        pred_cls_sorted = pred_cls[order]

        gt_matched = set()
        for i in range(len(tp)):
            pred_c = int(pred_cls_sorted[i])
            if tp[i]:
                gt_c = int(gt_cls[matched_gt[i]])
                cm[gt_c, pred_c] += 1
                gt_matched.add(matched_gt[i])
            else:
                cm[n - 1, pred_c] += 1  # FP

        for j in range(len(gt_cls)):
            if j not in gt_matched:
                cm[int(gt_cls[j]), n - 1] += 1  # FN

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict[int, str] | None = None,
    title: str = "Detection Confusion Matrix",
    normalize: bool = False,
) -> go.Figure:
    """Plot a Plotly heatmap for the detection confusion matrix."""
    names = class_names or CLASS_NAMES
    labels = [names.get(i, f"cls_{i}") for i in range(cm.shape[0] - 1)] + ["background"]

    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        data = np.round(data / row_sums, 3)

    text = [[str(int(v)) if not normalize else f"{v:.2f}" for v in row] for row in data]

    fig = go.Figure(go.Heatmap(
        z=data[::-1],
        x=labels,
        y=labels[::-1],
        text=text[::-1],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_white",
        height=550,
        width=600,
    )
    return fig


def per_class_metrics(cm: np.ndarray, class_names: Dict[int, str] | None = None) -> List[Dict]:
    """Derive precision, recall, F1 per class from the confusion matrix."""
    names = class_names or CLASS_NAMES
    n = cm.shape[0] - 1  # exclude background
    results = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        results.append({
            "class": names.get(i, f"cls_{i}"),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })
    return results
