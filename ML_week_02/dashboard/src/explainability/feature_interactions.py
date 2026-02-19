"""
feature_interactions.py – Detect and visualise pairwise feature interactions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator


def interaction_strength_h_statistic(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    grid_resolution: int = 20,
) -> float:
    """Approximate Friedman's H-statistic for the pair (a, b).

    H ≈ 0 → no interaction; H ≈ 1 → strong interaction.
    Uses a grid-based method on a random subsample.
    """
    from sklearn.inspection import partial_dependence

    cols = X.columns.tolist()
    idx_a, idx_b = cols.index(feature_a), cols.index(feature_b)
    sub = X.sample(min(500, len(X)), random_state=42)

    # PDP for pair
    pdp_ab = partial_dependence(estimator, sub, [(idx_a, idx_b)], grid_resolution=grid_resolution, kind="average")
    # PDP for each alone
    pdp_a = partial_dependence(estimator, sub, [idx_a], grid_resolution=grid_resolution, kind="average")
    pdp_b = partial_dependence(estimator, sub, [idx_b], grid_resolution=grid_resolution, kind="average")

    avg_ab = pdp_ab["average"][0]
    avg_a = pdp_a["average"][0]
    avg_b = pdp_b["average"][0]

    # H = var(PDP_ab - PDP_a - PDP_b) / var(PDP_ab)
    expected = avg_a[:, None] + avg_b[None, :]  # broadcasting
    interaction_part = avg_ab - expected

    var_inter = np.var(interaction_part)
    var_ab = np.var(avg_ab)

    return round(float(var_inter / var_ab) if var_ab > 0 else 0.0, 4)


def top_interactions(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    top_n: int = 10,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise H-statistics and return the top-N strongest pairs."""
    cols = features or X.select_dtypes("number").columns.tolist()
    # Limit to avoid combinatorial explosion
    cols = cols[:15]

    records = []
    for i, fa in enumerate(cols):
        for fb in cols[i + 1:]:
            try:
                h = interaction_strength_h_statistic(estimator, X, fa, fb, grid_resolution=15)
                records.append({"feature_a": fa, "feature_b": fb, "h_statistic": h})
            except Exception:
                continue

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("h_statistic", ascending=False).head(top_n).reset_index(drop=True)


def plot_interaction_heatmap(interactions_df: pd.DataFrame) -> go.Figure:
    """Heatmap of pairwise H-statistics."""
    if interactions_df.empty:
        return go.Figure()

    # Pivot to matrix
    features = sorted(set(interactions_df["feature_a"]) | set(interactions_df["feature_b"]))
    matrix = pd.DataFrame(0.0, index=features, columns=features)
    for _, row in interactions_df.iterrows():
        matrix.loc[row["feature_a"], row["feature_b"]] = row["h_statistic"]
        matrix.loc[row["feature_b"], row["feature_a"]] = row["h_statistic"]

    fig = px.imshow(
        matrix.values,
        x=features,
        y=features,
        color_continuous_scale="YlOrRd",
        text_auto=".3f",
        title="Feature Interaction Strength (H-statistic)",
        template="plotly_white",
    )
    fig.update_layout(height=500)
    return fig
