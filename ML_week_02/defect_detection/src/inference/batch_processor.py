"""
batch_processor.py – Batch / video processing pipeline.

Process full folders of images or video files frame-by-frame, saving
annotated outputs + a CSV summary.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.inference.predictor import DefectPredictor, PredictionResult
from src.utils.config import project_root
from src.utils.logging import get_logger
from src.utils.visualization import draw_detections

logger = get_logger(__name__)


class BatchProcessor:
    """Process images or video files at scale."""

    def __init__(
        self,
        predictor: DefectPredictor,
        output_dir: Path | None = None,
        save_annotated: bool = True,
    ):
        self.predictor = predictor
        self.output_dir = Path(output_dir) if output_dir else project_root() / "outputs" / "predictions"
        self.save_annotated = save_annotated
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Image folder processing ──────────────────────────────

    def process_folder(
        self,
        images_dir: Path,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[PredictionResult]:
        """Run detection on every image in *images_dir*.

        Returns a list of PredictionResult and writes a CSV summary.
        """
        images_dir = Path(images_dir)
        img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)
        logger.info(f"Batch: {len(img_paths)} images from {images_dir}")

        results: List[PredictionResult] = []
        csv_path = self.output_dir / "batch_results.csv"

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "n_detections", "inference_ms", "classes"])

            for img_path in img_paths:
                pred = self.predictor.predict_image(img_path)
                results.append(pred)

                if self.save_annotated:
                    self._save_annotated(img_path, pred)

                cls_summary = ", ".join(d.class_name for d in pred.detections)
                writer.writerow([
                    img_path.name, pred.count,
                    round(pred.inference_ms, 2), cls_summary,
                ])

        logger.success(f"Batch done — {len(results)} images → {csv_path}")
        return results

    # ── Video processing ─────────────────────────────────────

    def process_video(
        self,
        video_path: Path,
        stride: int = 1,
        max_frames: int | None = None,
        save_video: bool = True,
    ) -> List[PredictionResult]:
        """Run frame-by-frame detection on a video.

        Parameters
        ----------
        stride     : process every N-th frame
        max_frames : cap on total frames processed (None = all)
        save_video : write annotated output video
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {video_path.name} — {total} frames, {fps:.1f} FPS, {w}×{h}")

        writer = None
        if save_video:
            out_path = self.output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        results: List[PredictionResult] = []
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            if max_frames and processed >= max_frames:
                break

            pred = self.predictor.predict_image(frame)
            results.append(pred)

            if writer:
                annotated = self._annotate_frame(frame, pred)
                writer.write(annotated)

            processed += 1
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
            logger.success(f"Annotated video → {out_path}")

        logger.info(f"Processed {processed} / {total} frames")
        return results

    # ── Helpers ──────────────────────────────────────────────

    def _save_annotated(self, img_path: Path, pred: PredictionResult) -> None:
        image = cv2.imread(str(img_path))
        annotated = self._annotate_frame(image, pred)
        out_path = self.output_dir / f"{img_path.stem}_det{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

    @staticmethod
    def _annotate_frame(image: np.ndarray, pred: PredictionResult) -> np.ndarray:
        if not pred.detections:
            return image
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in pred.detections])
        classes = np.array([d.class_id for d in pred.detections])
        confs = np.array([d.confidence for d in pred.detections])
        return draw_detections(image, boxes, classes, confs)
