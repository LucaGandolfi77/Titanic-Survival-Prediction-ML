"""
synthetic_generator.py – Generate synthetic PCB defect images.

Defect types:
  0  scratch          – random line overlays with blur
  1  dent             – circular dark regions with gradient edges
  2  discoloration    – colour-shifted patches (oxidation / contamination)
  3  crack            – irregular branching lines (DLA / random walk)
  4  missing_component – rectangular mask-out

All bounding boxes are written in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>   (normalised 0-1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticDefectGenerator:
    """Generate synthetic PCB defects on clean background images (or plain boards)."""

    CLASS_NAMES = {0: "scratch", 1: "dent", 2: "discoloration", 3: "crack", 4: "missing_component"}

    def __init__(
        self,
        output_dir: Path,
        image_size: Tuple[int, int] = (640, 640),
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # ── Public entry point ───────────────────────────────────

    def generate_dataset(
        self,
        n_images: int = 2000,
        defects_per_image: Tuple[int, int] = (1, 4),
        background_dir: Path | None = None,
    ) -> Path:
        """Create a full synthetic dataset.

        If *background_dir* is given, random images from that folder are used
        as backgrounds; otherwise procedural PCB-like textures are generated.
        """
        bg_paths: list = []
        if background_dir and background_dir.exists():
            bg_paths = sorted(background_dir.glob("*.jpg")) + sorted(background_dir.glob("*.png"))

        logger.info(f"Generating {n_images} synthetic defect images → {self.output_dir}")

        for i in tqdm(range(n_images), desc="Generating"):
            img = self._get_background(bg_paths)
            n_def = self.rng.randint(defects_per_image[0], defects_per_image[1] + 1)
            annotations: List[Dict] = []

            for _ in range(n_def):
                defect_fn = self.rng.choice([
                    self._scratch, self._dent, self._discoloration,
                    self._crack, self._missing_component,
                ])
                img, ann = defect_fn(img)
                if ann is not None:
                    annotations.append(ann)

            # Save image
            img_path = self.output_dir / "images" / f"defect_{i:05d}.jpg"
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save label
            lbl_path = self.output_dir / "labels" / f"defect_{i:05d}.txt"
            with open(lbl_path, "w") as fh:
                for a in annotations:
                    bbox = a["bbox"]
                    fh.write(f"{a['class']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        logger.success(f"Generated {n_images} images in {self.output_dir}")
        return self.output_dir

    # ── Background ───────────────────────────────────────────

    def _get_background(self, paths: list) -> np.ndarray:
        if paths:
            p = self.rng.choice(paths)
            img = cv2.imread(str(p))
            return cv2.resize(img, self.image_size)
        return self._procedural_pcb()

    def _procedural_pcb(self) -> np.ndarray:
        """Generate a procedural PCB-like texture."""
        h, w = self.image_size
        # Dark-green base
        base = np.full((h, w, 3), (35, 80, 30), dtype=np.uint8)
        # Copper traces
        n_traces = self.rng.randint(15, 40)
        for _ in range(n_traces):
            pt1 = (self.rng.randint(0, w), self.rng.randint(0, h))
            pt2 = (self.rng.randint(0, w), self.rng.randint(0, h))
            colour = tuple(int(c) for c in self.rng.randint(150, 220, 3))
            cv2.line(base, pt1, pt2, colour, self.rng.randint(1, 4))
        # Components (rectangles)
        for _ in range(self.rng.randint(5, 15)):
            x, y = self.rng.randint(0, w - 30), self.rng.randint(0, h - 30)
            rw, rh = self.rng.randint(10, 50), self.rng.randint(6, 25)
            colour = tuple(int(c) for c in self.rng.randint(50, 180, 3))
            cv2.rectangle(base, (x, y), (x + rw, y + rh), colour, -1)
        # Gaussian noise
        noise = self.rng.normal(0, 5, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return base

    # ── Defect generators ────────────────────────────────────

    def _scratch(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        h, w = img.shape[:2]
        x0, y0 = self.rng.randint(0, w), self.rng.randint(0, h)
        length = self.rng.randint(40, min(w, h) // 2)
        angle = self.rng.uniform(0, 360)
        x1 = int(np.clip(x0 + length * np.cos(np.radians(angle)), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(np.radians(angle)), 0, h - 1))
        thick = self.rng.randint(1, 4)
        brightness = int(self.rng.randint(160, 230))

        out = img.copy()
        cv2.line(out, (x0, y0), (x1, y1), (brightness, brightness, brightness), thick)
        out = cv2.GaussianBlur(out, (3, 3), 0)

        bbox = self._xyxy_to_yolo(min(x0, x1) - thick, min(y0, y1) - thick,
                                   max(x0, x1) + thick, max(y0, y1) + thick, w, h)
        return out, {"class": 0, "bbox": bbox}

    def _dent(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        h, w = img.shape[:2]
        r = self.rng.randint(10, 35)
        cx, cy = self.rng.randint(r, w - r), self.rng.randint(r, h - r)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        blur = cv2.GaussianBlur(mask, (31, 31), 0) / 255.0
        dark = self.rng.uniform(0.3, 0.6)
        out = img.copy()
        for c in range(3):
            out[:, :, c] = (out[:, :, c] * (1 - blur * dark)).astype(np.uint8)
        bbox = self._xyxy_to_yolo(cx - r, cy - r, cx + r, cy + r, w, h)
        return out, {"class": 1, "bbox": bbox}

    def _discoloration(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        h, w = img.shape[:2]
        pw, ph = self.rng.randint(30, 80), self.rng.randint(30, 80)
        x, y = self.rng.randint(0, w - pw), self.rng.randint(0, h - ph)
        shifts = [
            np.array([0, 30, -20]),
            np.array([0, -20, 30]),
            np.array([-20, -20, 0]),
        ]
        shift = shifts[self.rng.randint(0, len(shifts))]
        out = img.copy()
        patch = np.clip(out[y:y + ph, x:x + pw].astype(np.int16) + shift, 0, 255).astype(np.uint8)
        mask = cv2.GaussianBlur(np.ones((ph, pw), np.float32), (15, 15), 0)
        for c in range(3):
            out[y:y + ph, x:x + pw, c] = (
                patch[:, :, c] * mask + out[y:y + ph, x:x + pw, c] * (1 - mask)
            ).astype(np.uint8)
        bbox = self._xyxy_to_yolo(x, y, x + pw, y + ph, w, h)
        return out, {"class": 2, "bbox": bbox}

    def _crack(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        h, w = img.shape[:2]
        pts = [(self.rng.randint(0, w), self.rng.randint(0, h))]
        for _ in range(self.rng.randint(5, 15)):
            dx, dy = self.rng.randint(-30, 30), self.rng.randint(-30, 30)
            nx = int(np.clip(pts[-1][0] + dx, 0, w - 1))
            ny = int(np.clip(pts[-1][1] + dy, 0, h - 1))
            pts.append((nx, ny))
        out = img.copy()
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (40, 40, 40), self.rng.randint(1, 3))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = self._xyxy_to_yolo(min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2, w, h)
        return out, {"class": 3, "bbox": bbox}

    def _missing_component(self, img: np.ndarray) -> Tuple[np.ndarray, Dict | None]:
        h, w = img.shape[:2]
        rw, rh = self.rng.randint(15, 55), self.rng.randint(10, 30)
        x, y = self.rng.randint(0, w - rw), self.rng.randint(0, h - rh)
        out = img.copy()
        # Fill with board colour + slight noise
        fill_col = np.array([30 + self.rng.randint(-5, 5),
                             75 + self.rng.randint(-10, 10),
                             28 + self.rng.randint(-5, 5)], dtype=np.uint8)
        out[y:y + rh, x:x + rw] = fill_col
        bbox = self._xyxy_to_yolo(x, y, x + rw, y + rh, w, h)
        return out, {"class": 4, "bbox": bbox}

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> List[float]:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)]
