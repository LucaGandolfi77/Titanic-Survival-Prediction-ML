"""
gradio_app.py – Gradio demo interface (alternative to Streamlit).

Launch:  python -m src.ui.gradio_app
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.predictor import DefectPredictor
from src.utils.config import project_root, CLASS_NAMES
from src.utils.visualization import draw_detections


def detect(image: np.ndarray, confidence: float = 0.25, iou: float = 0.45) -> tuple:
    """Run defect detection on an uploaded image.

    Returns (annotated_image, results_text).
    """
    trained = project_root() / "models" / "trained" / "best.pt"
    weights = trained if trained.exists() else "yolov8n.pt"
    predictor = DefectPredictor(weights=weights, device="cpu", conf=confidence, iou=iou)

    # Gradio provides RGB images
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = predictor.predict_image(bgr, conf=confidence)

    # Annotate
    if result.detections:
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in result.detections])
        classes = np.array([d.class_id for d in result.detections])
        confs = np.array([d.confidence for d in result.detections])
        annotated = draw_detections(bgr, boxes, classes, confs)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        annotated_rgb = image

    # Summary text
    lines = [f"**{result.count} defect(s) detected** ({result.inference_ms:.1f} ms)\n"]
    for d in result.detections:
        lines.append(f"- {d.class_name}: {d.confidence:.2%}")

    return annotated_rgb, "\n".join(lines)


def build_app():
    import gradio as gr

    demo = gr.Interface(
        fn=detect,
        inputs=[
            gr.Image(type="numpy", label="Upload PCB Image"),
            gr.Slider(0.05, 0.95, 0.25, step=0.05, label="Confidence Threshold"),
            gr.Slider(0.1, 0.9, 0.45, step=0.05, label="IoU Threshold"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Detections"),
            gr.Markdown(label="Results"),
        ],
        title="🔍 PCB Defect Detection",
        description="Upload a PCB image to detect manufacturing defects (scratch, dent, discoloration, crack, missing component).",
        examples=[],
        theme=gr.themes.Soft(),
    )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
