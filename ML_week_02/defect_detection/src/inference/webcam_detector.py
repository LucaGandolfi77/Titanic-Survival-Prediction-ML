"""
webcam_detector.py – Real-time detection from webcam / camera feed.

Opens a live OpenCV window, runs YOLOv8 frame-by-frame, and overlays
bounding boxes + FPS.  Press 'q' to quit, 's' to save a snapshot.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.inference.predictor import DefectPredictor
from src.utils.config import project_root
from src.utils.logging import get_logger
from src.utils.visualization import draw_detections

logger = get_logger(__name__)


class WebcamDetector:
    """Real-time defect detection from a camera stream."""

    def __init__(
        self,
        predictor: DefectPredictor,
        camera_id: int = 0,
        resolution: tuple = (1280, 720),
        show_fps: bool = True,
        record: bool = False,
        record_dir: Path | None = None,
    ):
        self.predictor = predictor
        self.camera_id = camera_id
        self.resolution = resolution
        self.show_fps = show_fps
        self.record = record
        self.record_dir = Path(record_dir) if record_dir else project_root() / "outputs" / "visualizations"
        self.record_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Open camera and start live detection loop.

        Controls:
          q  → quit
          s  → save current frame as snapshot
          +  → increase confidence threshold by 0.05
          -  → decrease confidence threshold by 0.05
        """
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            return

        writer = None
        if self.record:
            out_path = self.record_dir / "webcam_record.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 30, self.resolution)

        conf_threshold = self.predictor.conf
        frame_count = 0
        fps = 0.0
        t_prev = time.perf_counter()

        logger.info(f"Webcam started — camera {self.camera_id}, {self.resolution}")
        logger.info("Controls: q=quit, s=snapshot, +/-=conf threshold")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pred = self.predictor.predict_image(frame, conf=conf_threshold)

                # Draw detections
                if pred.detections:
                    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in pred.detections])
                    classes = np.array([d.class_id for d in pred.detections])
                    confs = np.array([d.confidence for d in pred.detections])
                    frame = draw_detections(frame, boxes, classes, confs)

                # FPS overlay
                frame_count += 1
                t_now = time.perf_counter()
                if t_now - t_prev >= 1.0:
                    fps = frame_count / (t_now - t_prev)
                    frame_count = 0
                    t_prev = t_now

                if self.show_fps:
                    cv2.putText(frame, f"FPS: {fps:.1f}  conf>={conf_threshold:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detections: {pred.count}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if writer:
                    writer.write(frame)

                cv2.imshow("Defect Detection — Live", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    snap_path = self.record_dir / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    logger.info(f"Snapshot → {snap_path}")
                elif key == ord("+") or key == ord("="):
                    conf_threshold = min(0.95, conf_threshold + 0.05)
                elif key == ord("-"):
                    conf_threshold = max(0.05, conf_threshold - 0.05)
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info("Webcam stopped.")
