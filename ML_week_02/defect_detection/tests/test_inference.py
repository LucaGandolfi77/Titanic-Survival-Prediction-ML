"""
test_inference.py – Tests for the detection predictor, metrics, and evaluation.

Tests that do NOT require a trained model or YOLO weights use pure-python
metric functions (compute_iou, match_predictions, evaluate_predictions).
Tests that require the YOLO runtime are marked ``@pytest.mark.yolo`` and
will be skipped when ultralytics is not installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────
# Detection / PredictionResult dataclasses
# ─────────────────────────────────────────────────────────────

class TestDetectionDataclass:
    def test_area(self):
        from src.inference.predictor import Detection
        d = Detection(x1=10, y1=20, x2=60, y2=70, confidence=0.9, class_id=0, class_name="scratch")
        assert d.area == pytest.approx(2500.0)

    def test_to_dict(self):
        from src.inference.predictor import Detection
        d = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.85, class_id=1, class_name="dent")
        out = d.to_dict()
        assert out["bbox"] == [0, 0, 10, 10]
        assert out["confidence"] == 0.85
        assert out["class_name"] == "dent"


class TestPredictionResult:
    def test_count(self):
        from src.inference.predictor import Detection, PredictionResult
        dets = [
            Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.8, class_id=0, class_name="scratch"),
            Detection(x1=5, y1=5, x2=15, y2=15, confidence=0.6, class_id=1, class_name="dent"),
        ]
        r = PredictionResult(detections=dets, inference_ms=12.5)
        assert r.count == 2

    def test_filter_by_confidence(self):
        from src.inference.predictor import Detection, PredictionResult
        dets = [
            Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0, class_name="scratch"),
            Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.3, class_id=1, class_name="dent"),
        ]
        r = PredictionResult(detections=dets, inference_ms=10.0)
        filtered = r.filter_by_confidence(0.5)
        assert filtered.count == 1
        assert filtered.detections[0].class_name == "scratch"

    def test_to_dict_structure(self):
        from src.inference.predictor import PredictionResult
        r = PredictionResult(detections=[], image_path="test.jpg", inference_ms=5.0)
        d = r.to_dict()
        assert d["image"] == "test.jpg"
        assert d["count"] == 0
        assert isinstance(d["detections"], list)


# ─────────────────────────────────────────────────────────────
# Evaluation metrics (pure numpy — no YOLO needed)
# ─────────────────────────────────────────────────────────────

class TestComputeIoU:
    def test_perfect_overlap(self):
        from src.evaluation.metrics import compute_iou
        box = np.array([0, 0, 10, 10])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.evaluation.metrics import compute_iou
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from src.evaluation.metrics import compute_iou
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        expected = 25.0 / (100 + 100 - 25)
        assert compute_iou(a, b) == pytest.approx(expected, rel=1e-4)

    def test_contained_box(self):
        from src.evaluation.metrics import compute_iou
        outer = np.array([0, 0, 20, 20])
        inner = np.array([5, 5, 10, 10])
        expected = 25.0 / (400 + 25 - 25)
        assert compute_iou(outer, inner) == pytest.approx(expected, rel=1e-4)


class TestComputeAP:
    def test_perfect_precision(self):
        from src.evaluation.metrics import compute_ap
        p = np.array([1.0, 1.0, 1.0, 1.0])
        r = np.array([0.25, 0.5, 0.75, 1.0])
        ap = compute_ap(p, r, method="interp11")
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_recall(self):
        from src.evaluation.metrics import compute_ap
        p = np.array([])
        r = np.array([])
        ap = compute_ap(p, r, method="all_points")
        assert ap == pytest.approx(0.0, abs=0.01)


class TestMatchPredictions:
    def test_perfect_match(self):
        from src.evaluation.metrics import match_predictions
        gt = np.array([[0, 0, 10, 10]])
        gt_cls = np.array([0])
        pred = np.array([[0, 0, 10, 10]])
        pred_cls = np.array([0])
        pred_conf = np.array([0.9])
        tp, fp, matched = match_predictions(gt, gt_cls, pred, pred_cls, pred_conf)
        assert tp[0] is True or tp[0] == True
        assert fp[0] is False or fp[0] == False

    def test_wrong_class(self):
        from src.evaluation.metrics import match_predictions
        gt = np.array([[0, 0, 10, 10]])
        gt_cls = np.array([0])
        pred = np.array([[0, 0, 10, 10]])
        pred_cls = np.array([1])  # different class
        pred_conf = np.array([0.9])
        tp, fp, _ = match_predictions(gt, gt_cls, pred, pred_cls, pred_conf)
        assert not tp[0]
        assert fp[0]

    def test_no_predictions(self):
        from src.evaluation.metrics import match_predictions
        gt = np.array([[0, 0, 10, 10]])
        gt_cls = np.array([0])
        pred = np.zeros((0, 4))
        pred_cls = np.array([], dtype=int)
        pred_conf = np.array([])
        tp, fp, _ = match_predictions(gt, gt_cls, pred, pred_cls, pred_conf)
        assert len(tp) == 0


class TestEvaluatePredictions:
    def test_perfect_mAP(self):
        from src.evaluation.metrics import evaluate_predictions
        gt_boxes = [np.array([[10, 10, 50, 50]])]
        gt_cls = [np.array([0])]
        pred_boxes = [np.array([[10, 10, 50, 50]])]
        pred_cls = [np.array([0])]
        pred_conf = [np.array([0.95])]
        result = evaluate_predictions(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
        assert "mAP50" in result
        assert result["mAP50"] > 0.9

    def test_empty_preds(self):
        from src.evaluation.metrics import evaluate_predictions
        gt_boxes = [np.array([[10, 10, 50, 50]])]
        gt_cls = [np.array([0])]
        pred_boxes = [np.zeros((0, 4))]
        pred_cls = [np.array([], dtype=int)]
        pred_conf = [np.array([])]
        result = evaluate_predictions(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
        assert result["mAP50"] == 0.0


# ─────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_build_matrix_shape(self):
        from src.evaluation.confusion_matrix import build_confusion_matrix
        gt_cls = [np.array([0, 1, 2])]
        pred_cls = [np.array([0, 1, 2])]
        gt_boxes = [np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])]
        pred_boxes = [np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])]
        pred_conf = [np.array([0.9, 0.8, 0.7])]
        cm = build_confusion_matrix(gt_cls, pred_cls, gt_boxes, pred_boxes, pred_conf, num_classes=5)
        assert cm.shape == (6, 6)  # 5 classes + 1 background

    def test_per_class_metrics(self):
        from src.evaluation.confusion_matrix import per_class_metrics
        cm = np.zeros((3, 3))
        cm[0, 0] = 5  # class 0: 5 TP
        cm[1, 1] = 3  # class 1: 3 TP
        cm[0, 1] = 2  # class 0 → class 1 misclassified
        metrics = per_class_metrics(cm, class_names={0: "scratch", 1: "dent"})
        assert "scratch" in [m["name"] for m in metrics]
        assert any(m["precision"] > 0 for m in metrics)
