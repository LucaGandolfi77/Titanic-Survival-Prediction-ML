"""
video_init_bbox.py – Utility to let the user select an initial bounding box
on the first frame of a video or webcam feed using OpenCV.

Usage
-----
    python -m src.demo.video_init_bbox --source path/to/video.mp4
    python -m src.demo.video_init_bbox  # webcam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2


def select_initial_bbox(source: str | int = 0) -> Tuple[Tuple[int, int, int, int], Path | None]:
    """Display the first frame and let the user draw a ROI.

    Returns
    -------
    bbox : (x, y, w, h)
    source_path : original source (or None for webcam)
    """
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame.")

    roi = cv2.selectROI("Select target — press ENTER to confirm", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    bbox = tuple(int(v) for v in roi)
    return bbox, Path(source) if isinstance(source, (str, Path)) and not str(source).isdigit() else None  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select initial bbox on first frame")
    parser.add_argument("--source", default="0", help="Video file or webcam index (default: 0)")
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    bbox, src_path = select_initial_bbox(src)
    print(f"Selected bbox (x, y, w, h): {bbox}")

    # Optionally save to annotation file
    out = Path(__file__).resolve().parents[3] / "data" / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    name = src_path.stem if src_path else "webcam"
    ann_file = out / f"{name}_init_bbox.json"
    ann_file.write_text(json.dumps({"bbox": list(bbox)}, indent=2))
    print(f"Saved to {ann_file}")


if __name__ == "__main__":
    main()
