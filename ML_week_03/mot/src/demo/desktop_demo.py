"""
desktop_demo.py – Real-time single-object tracking demo on macOS M1.

Opens the webcam (or a video file), lets the user draw the initial
bounding box, then tracks the object using the Siamese TFLite model.

Usage
-----
    python -m src.demo.desktop_demo                          # webcam
    python -m src.demo.desktop_demo --source path/video.mp4  # video file
    python -m src.demo.desktop_demo --model models/siamese_tracker.tflite
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.tracking.opencv_io import draw_bbox, draw_fps, open_video, select_roi
from src.tracking.tracker_core import SiameseTracker

ROOT = Path(__file__).resolve().parents[3]  # ML_week_03/mot


def run_demo(
    source: str | int = 0,
    model_path: Path | None = None,
    tracker_cfg: Path | None = None,
    model_cfg: Path | None = None,
) -> None:
    if model_path is None:
        model_path = ROOT / "models" / "siamese_tracker.tflite"
    if tracker_cfg is None:
        tracker_cfg = ROOT / "configs" / "tracker_config.yaml"
    if model_cfg is None:
        model_cfg = ROOT / "configs" / "model_config.yaml"

    cap = open_video(source)

    # ── read first frame & select ROI ─────────────────────────────
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")

    bbox = select_roi(first_frame, "Select target — press ENTER")
    if bbox[2] == 0 or bbox[3] == 0:
        print("No ROI selected. Exiting.")
        cap.release()
        return

    # ── initialise tracker ────────────────────────────────────────
    tracker = SiameseTracker(model_path, tracker_cfg, model_cfg)
    tracker.initialize(first_frame, bbox)

    print("Tracking started. Press ESC to quit.")
    fps = 0.0
    prev_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        new_bbox, score, latency_ms = tracker.track(frame)

        # FPS calculation (smoothed)
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        label = f"score={score:.2f}"
        draw_bbox(frame, new_bbox, label=label)
        draw_fps(frame, fps, latency_ms)

        cv2.imshow("Siamese Tracker — ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop tracking demo")
    parser.add_argument("--source", default="0", help="Video file or webcam index")
    parser.add_argument("--model", type=Path, default=None, help="TFLite model path")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_demo(src, args.model)


if __name__ == "__main__":
    main()
