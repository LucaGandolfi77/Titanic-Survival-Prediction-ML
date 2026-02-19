"""
visualization.py – Detection result visualisations for evaluation.

Functions return Plotly figures; no side-effects.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_detection_grid(
    images: List[np.ndarray],
    titles: List[str] | None = None,
    cols: int = 4,
) -> go.Figure:
    """Interactive image grid via Plotly."""
    from PIL import Image

    n = len(images)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles or [f"img_{i}" for i in range(n)])

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        pil_img = Image.fromarray(img[:, :, ::-1])  # BGR → RGB
        fig.add_trace(go.Image(z=np.array(pil_img)), row=r + 1, col=c + 1)

    fig.update_layout(height=300 * rows, template="plotly_white", showlegend=False)
    return fig


def plot_confidence_histogram(
    confidences: List[float],
    class_names: List[str] | None = None,
    class_ids: List[int] | None = None,
    title: str = "Confidence Distribution",
) -> go.Figure:
    """Histogram of detection confidence scores (optionally coloured by class)."""
    import pandas as pd

    data = {"confidence": confidences}
    if class_ids and class_names:
        name_map = {i: n for i, n in enumerate(class_names)}
        data["class"] = [name_map.get(c, str(c)) for c in class_ids]
        fig = px.histogram(pd.DataFrame(data), x="confidence", color="class",
                           nbins=30, title=title, template="plotly_white")
    else:
        fig = px.histogram(pd.DataFrame(data), x="confidence",
                           nbins=30, title=title, template="plotly_white")
    fig.update_layout(height=400)
    return fig


def plot_fps_timeline(
    inference_times_ms: List[float],
    title: str = "Inference Speed Timeline",
) -> go.Figure:
    """Line plot of FPS across frames."""
    fps = [1000.0 / t if t > 0 else 0 for t in inference_times_ms]
    fig = go.Figure(go.Scatter(
        y=fps, mode="lines",
        line=dict(color="#FF6B35"),
        name="FPS",
    ))
    fig.add_hline(y=np.mean(fps), line_dash="dash", line_color="blue",
                  annotation_text=f"avg {np.mean(fps):.1f} FPS")
    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title="FPS",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_mAP_comparison(
    models: Dict[str, Dict[str, float]],
    title: str = "Model Comparison — mAP",
) -> go.Figure:
    """Grouped bar chart comparing mAP50 and mAP50:95 across models."""
    names = list(models.keys())
    map50 = [models[n].get("mAP50", 0) for n in names]
    map5095 = [models[n].get("mAP50_95", 0) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=map50, name="mAP@0.5", marker_color="#4A90D9"))
    fig.add_trace(go.Bar(x=names, y=map5095, name="mAP@0.5:0.95", marker_color="#FF6B35"))
    fig.update_layout(
        title=title, barmode="group",
        yaxis_title="mAP", template="plotly_white", height=420,
    )
    return fig


def plot_per_class_ap(
    per_class: Dict[int, Dict],
    title: str = "Per-Class AP@0.5",
) -> go.Figure:
    """Horizontal bar chart of AP@0.5 per class."""
    names = [v["name"] for v in per_class.values()]
    aps = [v["AP50"] for v in per_class.values()]

    fig = px.bar(x=aps, y=names, orientation="h", title=title,
                 labels={"x": "AP@0.5", "y": "Class"}, template="plotly_white",
                 color=aps, color_continuous_scale="RdYlGn")
    fig.update_layout(height=max(300, 60 * len(names)), showlegend=False)
    return fig
