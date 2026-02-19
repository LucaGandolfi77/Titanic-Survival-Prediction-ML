"""
pdp.py – Partial Dependence Plots (PDP) and Individual Conditional
Expectation (ICE) curves.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence


def compute_pdp(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50,
    kind: str = "average",
) -> dict:
    """Compute PDP (and optionally ICE) for a single feature.

    Returns
    -------
    dict with keys: grid_values, average, individual (if kind='both')
    """
    feature_idx = list(X.columns).index(feature) if hasattr(X, "columns") else feature

    pdp_result = partial_dependence(
        estimator,
        X,
        features=[feature_idx],
        kind=kind,
        grid_resolution=grid_resolution,
    )

    result = {
        "grid_values": pdp_result["grid_values"][0],
        "average": pdp_result["average"][0] if "average" in pdp_result else pdp_result["average"][0],
    }

    if kind == "both" and "individual" in pdp_result:
        result["individual"] = pdp_result["individual"][0]

    return result


def compute_pdp_2d(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    grid_resolution: int = 30,
) -> dict:
    """Compute 2-D PDP for feature interaction."""
    cols = X.columns.tolist()
    idx_a, idx_b = cols.index(feature_a), cols.index(feature_b)

    pdp_result = partial_dependence(
        estimator,
        X,
        features=[(idx_a, idx_b)],
        kind="average",
        grid_resolution=grid_resolution,
    )

    return {
        "grid_a": pdp_result["grid_values"][0],
        "grid_b": pdp_result["grid_values"][1],
        "average": pdp_result["average"][0],
    }


def plot_pdp(
    pdp_data: dict,
    feature_name: str,
    title: str | None = None,
    show_ice: bool = False,
) -> go.Figure:
    """Interactive PDP line chart with optional ICE curves."""
    fig = go.Figure()

    # ICE curves
    if show_ice and "individual" in pdp_data:
        ice = pdp_data["individual"]
        for i in range(min(len(ice), 50)):
            fig.add_trace(go.Scatter(
                x=pdp_data["grid_values"],
                y=ice[i],
                mode="lines",
                line=dict(color="lightblue", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Average PDP
    fig.add_trace(go.Scatter(
        x=pdp_data["grid_values"],
        y=pdp_data["average"],
        mode="lines+markers",
        line=dict(color="#4A90D9", width=3),
        name="PDP (average)",
    ))

    fig.update_layout(
        title=title or f"Partial Dependence – {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Partial Dependence",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_pdp_2d(pdp_data: dict, feat_a: str, feat_b: str) -> go.Figure:
    """Contour plot for 2-D PDP."""
    fig = go.Figure(go.Contour(
        x=pdp_data["grid_a"],
        y=pdp_data["grid_b"],
        z=pdp_data["average"],
        colorscale="RdBu_r",
        colorbar=dict(title="PD"),
    ))
    fig.update_layout(
        title=f"2-D PDP – {feat_a} × {feat_b}",
        xaxis_title=feat_a,
        yaxis_title=feat_b,
        template="plotly_white",
        height=480,
    )
    return fig
