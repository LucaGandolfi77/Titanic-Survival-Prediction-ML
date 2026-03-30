"""
Pareto front visualization for multi-objective optimization results.

Produces both an interactive Plotly HTML figure and a static matplotlib
PNG for thesis inclusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pareto_front_matplotlib(
    pareto_data: List[Dict[str, Any]],
    title: str = "Pareto Front: F1 vs Training Time",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the Pareto front as a static matplotlib scatter plot.

    Args:
        pareto_data: List of dicts with 'f1' and 'training_time' keys.
        title: Plot title.
        save_path: If provided, save as PNG.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    f1_scores = [p["f1"] for p in pareto_data]
    times = [p["training_time"] for p in pareto_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(times, f1_scores, c=f1_scores, cmap="viridis",
                          s=60, edgecolors="black", linewidth=0.5, zorder=3)

    # Connect Pareto front points
    sorted_pairs = sorted(zip(times, f1_scores))
    ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
            "r--", alpha=0.5, linewidth=1, label="Pareto front")

    plt.colorbar(scatter, ax=ax, label="F1 Score")
    ax.set_xlabel("Training Time (s)", fontsize=12)
    ax.set_ylabel("F1 Score (macro)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pareto_front_plotly(
    pareto_data: List[Dict[str, Any]],
    title: str = "Pareto Front: F1 vs Training Time",
    save_path_html: Path | None = None,
    save_path_png: Path | None = None,
) -> Any:
    """Plot an interactive Pareto front using Plotly.

    Args:
        pareto_data: List of dicts with 'f1', 'training_time', 'description'.
        title: Plot title.
        save_path_html: If provided, save as interactive HTML.
        save_path_png: If provided, save as static PNG (requires kaleido).

    Returns:
        Plotly Figure object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed, skipping interactive Pareto front.")
        return None

    f1_scores = [p["f1"] for p in pareto_data]
    times = [p["training_time"] for p in pareto_data]
    descriptions = [p.get("description", "") for p in pareto_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=f1_scores,
        mode="markers+text",
        marker=dict(size=10, color=f1_scores, colorscale="Viridis",
                     showscale=True, colorbar=dict(title="F1")),
        text=[f"F1={f:.3f}" for f in f1_scores],
        textposition="top center",
        hovertext=descriptions,
        hoverinfo="text+x+y",
    ))

    sorted_pairs = sorted(zip(times, f1_scores))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in sorted_pairs],
        y=[p[1] for p in sorted_pairs],
        mode="lines",
        line=dict(color="red", dash="dash", width=1),
        name="Pareto front",
        showlegend=True,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Training Time (s)",
        yaxis_title="F1 Score (macro)",
        template="plotly_white",
        width=900, height=600,
    )

    if save_path_html is not None:
        save_path_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path_html))

    if save_path_png is not None:
        save_path_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(save_path_png), scale=2)
        except Exception:
            pass  # kaleido may not be available

    return fig


if __name__ == "__main__":
    fake_pareto = [
        {"f1": 0.95, "training_time": 2.0, "description": "Pipeline A"},
        {"f1": 0.93, "training_time": 1.0, "description": "Pipeline B"},
        {"f1": 0.90, "training_time": 0.5, "description": "Pipeline C"},
        {"f1": 0.88, "training_time": 0.3, "description": "Pipeline D"},
    ]
    plot_pareto_front_matplotlib(fake_pareto, title="Test Pareto")
    plt.show()
