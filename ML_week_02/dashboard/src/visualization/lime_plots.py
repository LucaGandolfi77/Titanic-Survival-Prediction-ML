"""
lime_plots.py – LIME explanation plots (Plotly).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go


# ── Local: LIME bar chart ─────────────────────────────────────

def plot_lime_explanation(
    feature_weights: List[Tuple[str, float]],
    prediction: float,
    prediction_proba: Optional[float] = None,
    top_k: int = 12,
    title: str = "LIME Local Explanation",
) -> go.Figure:
    """Horizontal bar chart from LIME feature weights."""
    fw = sorted(feature_weights, key=lambda x: abs(x[1]))[-top_k:]
    names = [n for n, _ in fw]
    vals = [v for _, v in fw]
    colors = ["#D9534F" if v > 0 else "#4A90D9" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
        )
    )
    subtitle = f"Pred: {prediction}"
    if prediction_proba is not None:
        subtitle += f" (prob: {prediction_proba:.3f})"

    fig.update_layout(
        title=f"{title}<br><sub>{subtitle}</sub>",
        xaxis_title="LIME weight",
        height=max(350, top_k * 28),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


# ── Side-by-side: SHAP vs LIME comparison ────────────────────

def plot_shap_vs_lime(
    shap_contributions: Dict[str, float],
    lime_contributions: Dict[str, float],
    top_k: int = 10,
    title: str = "SHAP vs LIME – Feature Attribution Comparison",
) -> go.Figure:
    """Side-by-side bar chart comparing SHAP and LIME attributions."""
    all_features = set(shap_contributions) | set(lime_contributions)
    df = pd.DataFrame({
        "feature": list(all_features),
        "SHAP": [shap_contributions.get(f, 0.0) for f in all_features],
        "LIME": [lime_contributions.get(f, 0.0) for f in all_features],
    })
    df["max_abs"] = df[["SHAP", "LIME"]].abs().max(axis=1)
    df = df.nlargest(top_k, "max_abs").sort_values("max_abs")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["SHAP"], name="SHAP",
        orientation="h", marker_color="#4A90D9",
    ))
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["LIME"], name="LIME",
        orientation="h", marker_color="#F5A623",
    ))
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Attribution",
        height=max(350, top_k * 35),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── LIME stability plot ──────────────────────────────────────

def plot_lime_stability(
    explanations: List[List[Tuple[str, float]]],
    top_k: int = 8,
    title: str = "LIME Explanation Stability",
) -> go.Figure:
    """Box-plot of LIME weights over multiple runs to show variance."""
    records = []
    for run_i, fw in enumerate(explanations):
        for feat, w in fw:
            records.append({"run": run_i, "feature": feat, "weight": w})

    df = pd.DataFrame(records)
    # Keep only features that appear most
    top_feats = df.groupby("feature")["weight"].apply(lambda s: s.abs().mean())
    top_feats = top_feats.nlargest(top_k).index.tolist()
    df = df[df["feature"].isin(top_feats)]

    fig = go.Figure()
    for feat in top_feats:
        subset = df[df["feature"] == feat]
        fig.add_trace(go.Box(y=subset["weight"], name=feat))

    fig.update_layout(
        title=title,
        yaxis_title="LIME weight",
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig
