"""
utils.py – Shared helpers for image warping, resizing, normalisation.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def crop_with_context(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    target_size: int,
    context_amount: float = 0.5,
) -> np.ndarray:
    """Crop a square patch centred on *bbox* with context padding.

    Parameters
    ----------
    image : (H, W, C) uint8
    bbox  : (x, y, w, h) — top-left corner + size
    target_size : output patch side length
    context_amount : extra context ratio

    Returns
    -------
    (target_size, target_size, 3) uint8 patch
    """
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2.0, y + h / 2.0
    wc = w + context_amount * (w + h)
    hc = h + context_amount * (w + h)
    size = int(round(np.sqrt(wc * hc)))
    size = max(size, 1)

    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x2 = x1 + size
    y2 = y1 + size

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - image.shape[0])
    pad_right = max(0, x2 - image.shape[1])

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        avg_color = image.mean(axis=(0, 1)).astype(np.uint8).tolist()
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=avg_color,
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    crop = image[max(0, y1): y2, max(0, x1): x2]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def normalise(patch: np.ndarray) -> np.ndarray:
    """Convert uint8 → float32 [0, 1]."""
    return patch.astype(np.float32) / 255.0


def cosine_window(h: int, w: int) -> np.ndarray:
    """Create a (h, w) cosine window (Hanning) for score penalisation."""
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return np.outer(wy, wx)


def clip_bbox(
    bbox: Tuple[float, float, float, float],
    img_h: int,
    img_w: int,
) -> Tuple[float, float, float, float]:
    """Clip bbox to image boundaries."""
    x, y, w, h = bbox
    x = max(0.0, min(x, img_w - 1))
    y = max(0.0, min(y, img_h - 1))
    w = max(1.0, min(w, img_w - x))
    h = max(1.0, min(h, img_h - y))
    return (x, y, w, h)
