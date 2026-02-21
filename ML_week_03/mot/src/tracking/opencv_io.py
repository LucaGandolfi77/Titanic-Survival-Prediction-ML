"""
opencv_io.py – Video capture, ROI extraction, bounding-box drawing via OpenCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


def open_video(source: str | int | Path = 0) -> cv2.VideoCapture:
    """Open a video file or webcam.

    Parameters
    ----------
    source : file path or device index (``0`` for default webcam).
    """
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def frame_generator(
    cap: cv2.VideoCapture,
) -> Generator[np.ndarray, None, None]:
    """Yield BGR frames from an open ``VideoCapture``."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    font_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw a bounding box (x, y, w, h) and optional label on *frame* (in-place)."""
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
        )
    return frame


def draw_fps(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Overlay FPS and latency text on the top-left of *frame*."""
    text = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame


def select_roi(frame: np.ndarray, window_name: str = "Select ROI") -> Tuple[int, int, int, int]:
    """Show *frame* and let the user draw a ROI rectangle.

    Returns (x, y, w, h).
    """
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return tuple(int(v) for v in roi)  # type: ignore[return-value]
