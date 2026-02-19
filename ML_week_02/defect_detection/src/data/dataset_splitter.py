"""
dataset_splitter.py – Stratified train / val / test split for YOLO datasets.

Ensures each split has a representative distribution of defect classes.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def stratified_split(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> Dict[str, int]:
    """Split a YOLO dataset into train / val / test with class-stratification.

    Parameters
    ----------
    images_dir : folder with .jpg images
    labels_dir : folder with matching .txt YOLO labels
    output_dir : create train/, val/, test/ sub-dirs here
    ratios     : (train, val, test) proportions  – must sum to 1

    Returns
    -------
    dict mapping split name → number of images
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1, got {sum(ratios)}"

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    logger.info(f"Found {len(img_paths)} images")

    # Read primary class per image (the first annotation's class)
    classes: List[int] = []
    valid_paths: List[Path] = []
    for img in img_paths:
        lbl = labels_dir / img.with_suffix(".txt").name
        if not lbl.exists():
            classes.append(-1)  # no label → "background"
        else:
            first_line = lbl.read_text().strip().split("\n")[0]
            classes.append(int(first_line.split()[0]) if first_line else -1)
        valid_paths.append(img)

    classes_arr = np.array(classes)
    rng = np.random.RandomState(seed)

    # Stratify by class
    indices = np.arange(len(valid_paths))
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(classes_arr):
        cls_indices = indices[classes_arr == cls]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    counts = {}

    for split_name, idxs in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i in idxs:
            src_img = valid_paths[i]
            src_lbl = labels_dir / src_img.with_suffix(".txt").name
            shutil.copy2(src_img, img_out / src_img.name)
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_out / src_lbl.name)
        counts[split_name] = len(idxs)

    logger.success(f"Split complete → train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts


def compute_class_distribution(labels_dir: Path) -> Dict[int, int]:
    """Count instances of each class in a YOLO label directory."""
    counter: Counter = Counter()
    for lbl in sorted(Path(labels_dir).glob("*.txt")):
        for line in lbl.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                counter[int(parts[0])] += 1
    return dict(sorted(counter.items()))
