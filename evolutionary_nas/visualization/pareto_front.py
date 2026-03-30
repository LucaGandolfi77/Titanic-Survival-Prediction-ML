"""
Pareto Front
=============
2D scatter: accuracy vs parameter count. Plotly interactive + static PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pareto_front_matplotlib(
    pareto_data: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front: Accuracy vs Parameters",
) -> None:
    """Static Pareto front scatter plot."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    accs = [d["accuracy"] for d in pareto_data]
    params = [d["param_count"] for d in pareto_data]

    sorted_idx = np.argsort(params)
    sorted_params = [params[i] for i in sorted_idx]
    sorted_accs = [accs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(params, accs, s=60, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3)
    ax.plot(sorted_params, sorted_accs, "r--", alpha=0.5, linewidth=1.5,
            label="Pareto front", zorder=2)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_front_plotly(
    pareto_data: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front: Accuracy vs Parameters",
) -> None:
    """Interactive Plotly Pareto front with hover showing architecture."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    accs = [d["accuracy"] for d in pareto_data]
    params = [d["param_count"] for d in pareto_data]
    descs = [d.get("description", "") for d in pareto_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=params, y=accs, mode="markers",
        marker=dict(size=10, color=accs, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Accuracy")),
        text=descs, hoverinfo="text+x+y",
        name="Architectures",
    ))

    sorted_idx = np.argsort(params)
    fig.add_trace(go.Scatter(
        x=[params[i] for i in sorted_idx],
        y=[accs[i] for i in sorted_idx],
        mode="lines", line=dict(color="red", dash="dash"),
        name="Pareto front",
    ))

    fig.update_layout(
        title=title, xaxis_title="Parameter Count",
        yaxis_title="Accuracy", template="plotly_white",
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        html_path = save_path.with_suffix(".html")
        fig.write_html(str(html_path))
        try:
            fig.write_image(str(save_path), scale=2)
        except Exception:
            pass


if __name__ == "__main__":
    data = [
        {"accuracy": 0.85 + i * 0.01, "param_count": 10000 + i * 5000,
         "description": f"Arch {i}"}
        for i in range(20)
    ]
    plot_pareto_front_matplotlib(data, Path("/tmp/pareto.png"))
    print("Pareto plot saved.")
