"""
test_augmentation.py – Tests for the augmentation pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def dummy_image():
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_bboxes():
    """list of YOLO-format bboxes [[cx, cy, w, h], ...]."""
    return [[0.5, 0.5, 0.2, 0.3], [0.2, 0.8, 0.1, 0.1]]


@pytest.fixture()
def dummy_labels():
    return [0, 2]


@pytest.fixture()
def label_files(tmp_path: Path):
    """Create tiny image-label pairs on disk for offline augmentation."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for i in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
        (lbl_dir / f"img_{i:04d}.txt").write_text(f"0 0.5 0.5 0.2 0.2\n")
    return img_dir, lbl_dir


# ─────────────────────────────────────────────────────────────
# Pipeline construction
# ─────────────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_default_pipeline(self):
        from src.data.augmentation import build_augmentation_pipeline

        pipe = build_augmentation_pipeline()
        assert pipe is not None
        assert hasattr(pipe, "transforms")

    def test_pipeline_from_yaml(self, tmp_path: Path):
        import yaml
        from src.data.augmentation import build_augmentation_pipeline

        cfg = {
            "albumentations": {
                "transforms": [
                    {"name": "HorizontalFlip", "p": 0.5},
                    {"name": "RandomBrightnessContrast", "p": 0.3},
                ]
            }
        }
        cfg_path = tmp_path / "aug.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        pipe = build_augmentation_pipeline(config_path=cfg_path)
        assert len(pipe.transforms) >= 2

    def test_missing_config_returns_defaults(self):
        from src.data.augmentation import build_augmentation_pipeline

        pipe = build_augmentation_pipeline(config_path=Path("/nonexistent.yaml"))
        assert len(pipe.transforms) >= 5  # defaults have many transforms


# ─────────────────────────────────────────────────────────────
# Image + bbox augmentation
# ─────────────────────────────────────────────────────────────

class TestAugmentImageAndBoxes:
    def test_returns_three_items(self, dummy_image, dummy_bboxes, dummy_labels):
        from src.data.augmentation import augment_image_and_boxes, build_augmentation_pipeline

        pipe = build_augmentation_pipeline()
        aug_img, aug_bboxes, aug_labels = augment_image_and_boxes(
            dummy_image, dummy_bboxes, dummy_labels, pipe
        )
        assert isinstance(aug_img, np.ndarray)
        assert isinstance(aug_bboxes, (list, tuple))
        assert isinstance(aug_labels, (list, tuple))

    def test_preserves_label_count(self, dummy_image, dummy_bboxes, dummy_labels):
        """Label count can decrease (out-of-frame) but never increase."""
        from src.data.augmentation import augment_image_and_boxes, build_augmentation_pipeline

        pipe = build_augmentation_pipeline()
        _, _, aug_labels = augment_image_and_boxes(
            dummy_image, dummy_bboxes, dummy_labels, pipe
        )
        assert len(aug_labels) <= len(dummy_labels)


# ─────────────────────────────────────────────────────────────
# Offline augmentation
# ─────────────────────────────────────────────────────────────

class TestOfflineAugment:
    def test_offline_augment_count(self, label_files, tmp_path: Path):
        from src.data.augmentation import build_augmentation_pipeline, offline_augment_dataset

        # Skip Normalize for offline augmentation (reads as uint8, would fail)
        import albumentations as A
        pipe = A.Compose(
            [A.HorizontalFlip(p=0.5)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )
        img_dir, lbl_dir = label_files
        out_img = tmp_path / "aug_images"
        out_lbl = tmp_path / "aug_labels"
        n = offline_augment_dataset(img_dir, lbl_dir, out_img, out_lbl, pipeline=pipe, factor=2)
        assert n == 10  # 5 images × factor 2
        assert len(list(out_img.glob("*.jpg"))) == 10
        assert len(list(out_lbl.glob("*.txt"))) == 10

    def test_augmented_labels_valid(self, label_files, tmp_path: Path):
        import albumentations as A
        from src.data.augmentation import offline_augment_dataset

        pipe = A.Compose(
            [A.HorizontalFlip(p=0.5)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )
        img_dir, lbl_dir = label_files
        out_img = tmp_path / "aug_img2"
        out_lbl = tmp_path / "aug_lbl2"
        offline_augment_dataset(img_dir, lbl_dir, out_img, out_lbl, pipeline=pipe, factor=1)
        for lbl_f in out_lbl.glob("*.txt"):
            for line in lbl_f.read_text().strip().split("\n"):
                parts = line.split()
                assert len(parts) == 5
                assert 0 <= float(parts[1]) <= 1
                assert 0 <= float(parts[2]) <= 1


# ─────────────────────────────────────────────────────────────
# YOLO label I/O
# ─────────────────────────────────────────────────────────────

class TestLabelIO:
    def test_read_write_roundtrip(self, tmp_path: Path):
        from src.data.augmentation import _read_yolo_labels, _write_yolo_labels

        lbl = tmp_path / "test.txt"
        original_bboxes = [[0.5, 0.5, 0.2, 0.3]]
        original_classes = [2]
        _write_yolo_labels(lbl, original_bboxes, original_classes)

        bboxes, classes = _read_yolo_labels(lbl)
        assert classes == [2]
        assert len(bboxes) == 1
        np.testing.assert_allclose(bboxes[0], [0.5, 0.5, 0.2, 0.3], atol=1e-5)
