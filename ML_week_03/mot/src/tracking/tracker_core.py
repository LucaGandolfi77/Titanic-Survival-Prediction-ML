"""
tracker_core.py – Single Object Tracker using the Siamese TFLite model.

This module is **independent of training code**: it only needs the TFLite
model file and the YAML config files.

Usage
-----
    from src.tracking.tracker_core import SiameseTracker

    tracker = SiameseTracker(model_path, tracker_config_path, model_config_path)
    tracker.initialize(first_frame, bbox)
    for frame in video:
        bbox, score, latency_ms = tracker.track(frame)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from scipy.special import expit as sigmoid  # type: ignore[import-untyped]

from src.tracking.tflite_inference import TFLiteTracker
from src.tracking.utils import clip_bbox, cosine_window, crop_with_context, normalise


class SiameseTracker:
    """Siamese single-object tracker backed by a TFLite model.

    Parameters
    ----------
    model_path : path to ``siamese_tracker.tflite``
    tracker_cfg : path to ``tracker_config.yaml``
    model_cfg   : path to ``model_config.yaml``
    """

    def __init__(
        self,
        model_path: Path | str,
        tracker_cfg: Path | str | None = None,
        model_cfg: Path | str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[3]

        if tracker_cfg is None:
            tracker_cfg = root / "configs" / "tracker_config.yaml"
        if model_cfg is None:
            model_cfg = root / "configs" / "model_config.yaml"

        self._tcfg = yaml.safe_load(Path(tracker_cfg).read_text())["tracker"]
        self._mcfg = yaml.safe_load(Path(model_cfg).read_text())

        self._template_size: int = self._mcfg["template"]["size"]
        self._search_size: int = self._mcfg["search"]["size"]
        self._context: float = self._mcfg["template"]["context_amount"]
        self._window_influence: float = self._tcfg["scoring"]["window_influence"]
        self._ema_decay: float = self._tcfg["template_update"].get("ema_decay", 0.95)
        self._update_template: bool = self._tcfg["template_update"].get("enabled", True)

        self._engine = TFLiteTracker(model_path)

        # state (set on initialize)
        self._bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._template_patch: Optional[np.ndarray] = None
        self._cos_window: Optional[np.ndarray] = None

    # ────────────────── public API ────────────────────────────────
    def initialize(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> None:
        """Initialise tracker with the first frame and target bounding box (x, y, w, h)."""
        self._bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        self._template_patch = self._get_template(frame)

    def track(self, frame: np.ndarray) -> Tuple[Tuple[float, float, float, float], float, float]:
        """Track the target in *frame*.

        Returns
        -------
        bbox       : updated (x, y, w, h)
        best_score : peak response score
        latency_ms : inference time
        """
        if self._template_patch is None:
            raise RuntimeError("Call initialize() before track().")

        search_patch = crop_with_context(
            frame, self._bbox, self._search_size, self._context,
        )
        search_norm = normalise(search_patch)[np.newaxis]  # (1, S, S, 3)
        template_norm = normalise(self._template_patch)[np.newaxis]  # (1, T, T, 3)

        response_map, latency_ms = self._engine.infer(template_norm, search_norm)
        response_map = sigmoid(response_map)  # to [0, 1]

        # ── apply cosine window penalty ───────────────────────────
        rh, rw = response_map.shape
        if self._cos_window is None or self._cos_window.shape != (rh, rw):
            self._cos_window = cosine_window(rh, rw)
        penalised = (1 - self._window_influence) * response_map + \
                    self._window_influence * self._cos_window

        # ── locate peak ───────────────────────────────────────────
        best_idx = np.unravel_index(np.argmax(penalised), penalised.shape)
        best_score = float(response_map[best_idx])

        # Map peak back to image coordinates
        x, y, w, h = self._bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Response map centre
        rc_y, rc_x = rh / 2.0, rw / 2.0
        # Pixel shift in response map
        dy = (best_idx[0] - rc_y)
        dx = (best_idx[1] - rc_x)
        # Scale shift to search-region pixels  (stride ≈ search / response)
        stride = self._search_size / max(rh, 1)
        cx += dx * stride * (w / self._search_size)
        cy += dy * stride * (h / self._search_size)

        new_bbox = clip_bbox((cx - w / 2.0, cy - h / 2.0, w, h), frame.shape[0], frame.shape[1])
        self._bbox = new_bbox

        # ── optional template update (EMA) ────────────────────────
        if self._update_template:
            new_tmpl = crop_with_context(frame, self._bbox, self._template_size, self._context)
            alpha = self._ema_decay
            self._template_patch = cv2.addWeighted(
                self._template_patch, alpha, new_tmpl, 1 - alpha, 0,
            )

        return self._bbox, best_score, latency_ms

    # ────────────────── helpers ───────────────────────────────────
    def _get_template(self, frame: np.ndarray) -> np.ndarray:
        return crop_with_context(frame, self._bbox, self._template_size, self._context)
