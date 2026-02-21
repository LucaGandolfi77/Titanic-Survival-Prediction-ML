"""
dataset.py – Generate (template, search, label) pairs for Siamese tracker training.

Works on any dataset structured as:
    data/raw/<sequence_name>/
        imgs/     ← frame images (001.jpg, 002.jpg, …)
        groundtruth.txt   ← per-frame bounding boxes  x,y,w,h

Also supports **synthetic** pair generation for quick smoke-testing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


# ────────────────────────── helpers ───────────────────────────────────
def _crop_and_resize(
    image: np.ndarray,
    bbox: np.ndarray,
    target_size: int,
    context_amount: float = 0.5,
) -> np.ndarray:
    """Crop centred patch around *bbox* with context, resize to *target_size*.

    Parameters
    ----------
    image : HWC uint8 array
    bbox  : (x, y, w, h)
    """
    x, y, w, h = bbox.astype(np.float32)
    cx, cy = x + w / 2, y + h / 2
    wc = w + context_amount * (w + h)
    hc = h + context_amount * (w + h)
    size = int(round(np.sqrt(wc * hc)))

    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x2 = x1 + size
    y2 = y1 + size

    # pad if out of bounds
    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - image.shape[0])
    pad_right = max(0, x2 - image.shape[1])

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        avg_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=avg_color.tolist(),
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    patch = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return patch


def _parse_groundtruth(gt_path: Path) -> np.ndarray:
    """Read ground-truth file → (N, 4) array of (x, y, w, h)."""
    lines = gt_path.read_text().strip().splitlines()
    bboxes: list[list[float]] = []
    for line in lines:
        parts = line.replace("\t", ",").replace(" ", ",").split(",")
        parts = [p for p in parts if p]
        bboxes.append([float(v) for v in parts[:4]])
    return np.array(bboxes, dtype=np.float32)


# ────────────────────── data augmentation ─────────────────────────────
def _augment_patch(patch: np.ndarray, cfg: Dict) -> np.ndarray:
    """Apply random augmentations to a patch (HWC uint8)."""
    if cfg.get("h_flip", False) and random.random() < 0.5:
        patch = cv2.flip(patch, 1)

    if cfg.get("color_jitter", False):
        # brightness
        bmin, bmax = cfg.get("brightness_range", [0.8, 1.2])
        factor = random.uniform(bmin, bmax)
        patch = np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        # contrast
        cmin, cmax = cfg.get("contrast_range", [0.8, 1.2])
        factor = random.uniform(cmin, cmax)
        mean = patch.mean()
        patch = np.clip((patch.astype(np.float32) - mean) * factor + mean, 0, 255).astype(
            np.uint8
        )

    if cfg.get("random_erase", False) and random.random() < 0.3:
        h, w = patch.shape[:2]
        eh, ew = random.randint(h // 8, h // 4), random.randint(w // 8, w // 4)
        ey, ex = random.randint(0, h - eh), random.randint(0, w - ew)
        patch[ey : ey + eh, ex : ex + ew] = 0

    return patch


# ─────────────────── sequence-level pair generation ───────────────────
def _generate_pairs_from_sequence(
    seq_dir: Path,
    template_size: int = 127,
    search_size: int = 255,
    max_pairs: int = 50,
    context_amount: float = 0.5,
    augmentations: Optional[Dict] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (template, search, label_map) tuples from one video sequence."""
    imgs_dir = seq_dir / "imgs"
    gt_path = seq_dir / "groundtruth.txt"
    if not imgs_dir.exists() or not gt_path.exists():
        return []

    frames_sorted = sorted(imgs_dir.glob("*.*"))
    bboxes = _parse_groundtruth(gt_path)
    n = min(len(frames_sorted), len(bboxes))
    if n < 2:
        return []

    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    indices = list(range(n))
    random.shuffle(indices)

    for idx in indices[: max_pairs]:
        # template from frame idx, search from a nearby frame
        gap = random.randint(1, min(10, n - 1))
        search_idx = min(idx + gap, n - 1)
        if search_idx == idx:
            search_idx = max(0, idx - gap)

        img_t = cv2.imread(str(frames_sorted[idx]))
        img_s = cv2.imread(str(frames_sorted[search_idx]))
        if img_t is None or img_s is None:
            continue

        template_patch = _crop_and_resize(img_t, bboxes[idx], template_size, context_amount)
        search_patch = _crop_and_resize(img_s, bboxes[search_idx], search_size, context_amount)

        if augmentations:
            search_patch = _augment_patch(search_patch, augmentations)

        # Ground-truth label: Gaussian centred at target position in response map
        # Approximate response map size based on model stride
        resp_h = (search_size - template_size) // 8 + 1
        resp_w = resp_h
        label = _make_gaussian_label(resp_h, resp_w)

        pairs.append((template_patch, search_patch, label))
    return pairs


def _make_gaussian_label(h: int, w: int, sigma: float = 2.0) -> np.ndarray:
    """Create a (h, w, 1) Gaussian label centred in the response map."""
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    label = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return label[..., np.newaxis]


# ────────────── synthetic pairs (for smoke test / no dataset) ─────────
def generate_synthetic_pairs(
    n_pairs: int = 500,
    template_size: int = 127,
    search_size: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random noise pairs for quick pipeline validation."""
    templates = np.random.randint(0, 256, (n_pairs, template_size, template_size, 3), dtype=np.uint8)
    searches = np.random.randint(0, 256, (n_pairs, search_size, search_size, 3), dtype=np.uint8)
    resp_h = (search_size - template_size) // 8 + 1
    labels = np.stack([_make_gaussian_label(resp_h, resp_h) for _ in range(n_pairs)])
    return templates, searches, labels


# ──────────────── tf.data pipeline ────────────────────────────────────
def build_tf_dataset(
    data_dir: Path,
    template_size: int = 127,
    search_size: int = 255,
    max_pairs_per_video: int = 50,
    augmentations: Optional[Dict] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` yielding ((template, search), label) batches.

    If *data_dir* contains no sequences, falls back to synthetic data.
    """
    all_templates: List[np.ndarray] = []
    all_searches: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for seq_dir in sorted(raw_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            pairs = _generate_pairs_from_sequence(
                seq_dir,
                template_size=template_size,
                search_size=search_size,
                max_pairs=max_pairs_per_video,
                augmentations=augmentations,
            )
            for t, s, l in pairs:
                all_templates.append(t)
                all_searches.append(s)
                all_labels.append(l)

    if len(all_templates) == 0:
        print("[dataset] No real data found — generating synthetic pairs for smoke test.")
        t_arr, s_arr, l_arr = generate_synthetic_pairs(500, template_size, search_size)
    else:
        t_arr = np.stack(all_templates)
        s_arr = np.stack(all_searches)
        l_arr = np.stack(all_labels)

    # Normalise to [0, 1]
    t_arr = t_arr.astype(np.float32) / 255.0
    s_arr = s_arr.astype(np.float32) / 255.0

    ds = tf.data.Dataset.from_tensor_slices(((t_arr, s_arr), l_arr))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(t_arr), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
