"""
augmentation.py – Custom Albumentations augmentation pipeline for PCB images.

Provides a factory that builds an ``A.Compose`` pipeline from a config dict
or YAML file.  Handles both bounding-box-aware and image-only pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import yaml


def build_augmentation_pipeline(
    config_path: Optional[Path] = None,
    bbox_format: str = "yolo",
    min_visibility: float = 0.3,
) -> A.Compose:
    """Build an Albumentations pipeline from a YAML config.

    If *config_path* is None, returns a sensible default pipeline.
    """
    if config_path and config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        transforms = _parse_transforms(cfg.get("albumentations", {}).get("transforms", []))
    else:
        transforms = _default_transforms()

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format=bbox_format,
            label_fields=["class_labels"],
            min_visibility=min_visibility,
        ),
    )


def _default_transforms() -> List[A.BasicTransform]:
    """Sensible defaults for PCB defect detection."""
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]


def _parse_transforms(transform_list: List[Dict[str, Any]]) -> List[A.BasicTransform]:
    """Instantiate Albumentations transforms from config dicts."""
    mapping = {
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "GaussianBlur": A.GaussianBlur,
        "GaussNoise": A.GaussNoise,
        "CLAHE": A.CLAHE,
        "CoarseDropout": A.CoarseDropout,
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "RandomRotate90": A.RandomRotate90,
        "Normalize": A.Normalize,
    }
    transforms: List[A.BasicTransform] = []
    for entry in transform_list:
        name = entry.pop("name", None)
        cls = mapping.get(name)
        if cls is None:
            continue
        transforms.append(cls(**entry))
    return transforms


def augment_image_and_boxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    pipeline: A.Compose,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Apply the augmentation *pipeline* to an image + its YOLO bboxes."""
    result = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return result["image"], result["bboxes"], result["class_labels"]


def offline_augment_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images: Path,
    output_labels: Path,
    pipeline: A.Compose,
    factor: int = 3,
) -> int:
    """Create *factor* augmented copies for every image-label pair.

    Returns the total number of augmented images written.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        lbl_path = labels_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        bboxes, class_labels = _read_yolo_labels(lbl_path)

        for j in range(factor):
            aug_img, aug_bboxes, aug_cls = augment_image_and_boxes(image, bboxes, class_labels, pipeline)
            stem = f"{img_path.stem}_aug{j}"
            cv2.imwrite(str(output_images / f"{stem}.jpg"), aug_img)
            _write_yolo_labels(output_labels / f"{stem}.txt", aug_bboxes, aug_cls)
            written += 1

    return written


# ── I/O helpers ──────────────────────────────────────────────

def _read_yolo_labels(path: Path) -> Tuple[List[List[float]], List[int]]:
    bboxes, classes = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            classes.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, classes


def _write_yolo_labels(path: Path, bboxes: List, classes: List[int]) -> None:
    with open(path, "w") as fh:
        for cls, box in zip(classes, bboxes):
            fh.write(f"{cls} " + " ".join(f"{v:.6f}" for v in box) + "\n")
