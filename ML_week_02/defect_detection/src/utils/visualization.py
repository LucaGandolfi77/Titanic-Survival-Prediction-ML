"""
visualization.py – General-purpose plotting utilities.

All functions return Plotly figures or numpy images; no side-effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import class_colours, CLASS_NAMES


# ── Drawing ──────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    class_names: Dict[int, str] | None = None,
    line_width: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes + labels on *image* (BGR).

    Parameters
    ----------
    boxes : (N, 4) array of [x1, y1, x2, y2]
    classes : (N,) integer class indices
    confidences : (N,) floats 0‒1
    """
    cmap = class_colours()
    names = class_names or CLASS_NAMES
    img = image.copy()

    for box, cls_id, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        name = names.get(int(cls_id), f"cls_{cls_id}")
        colour = cmap.get(name, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, line_width)

        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ── Plotly helpers ───────────────────────────────────────────

def plot_class_distribution(counts: Dict[str, int], title: str = "Class Distribution") -> go.Figure:
    """Bar chart of class counts."""
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        color=list(counts.keys()),
        title=title,
        labels={"x": "Defect Class", "y": "Count"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Curves",
) -> go.Figure:
    """Line plot for loss / mAP curves across epochs."""
    fig = go.Figure()
    for name, values in metrics.items():
        fig.add_trace(go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines+markers",
            name=name,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_pr_curve(
    precision: Sequence[float],
    recall: Sequence[float],
    ap: float,
    class_name: str = "all",
) -> go.Figure:
    """Precision-Recall curve."""
    fig = go.Figure(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        fill="tozeroy",
        name=f"{class_name} (AP={ap:.3f})",
    ))
    fig.update_layout(
        title=f"Precision-Recall – {class_name}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=420,
    )
    return fig


def image_grid(
    images: List[np.ndarray],
    cols: int = 4,
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Tile a list of images into a single grid image."""
    rows_needed = (len(images) + cols - 1) // cols
    resized = [cv2.resize(img, size) for img in images]
    # Pad to fill grid
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((*size[::-1], 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows_needed):
        row_imgs = resized[r * cols: (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)
