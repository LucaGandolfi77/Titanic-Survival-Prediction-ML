"""
decision_boundary.py – 2-D decision boundary visualization via PCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


def decision_boundary_2d(
    estimator: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray | None = None,
    y_test: pd.Series | np.ndarray | None = None,
    resolution: int = 200,
    title: str = "Decision Boundary (PCA 2-D)",
) -> go.Figure:
    """Project features to 2-D with PCA, then draw the decision surface.

    This is approximate because we retrain a lightweight clone on the
    2-D PCA projection – it gives a visual intuition, not exact boundaries.
    """
    from sklearn.base import clone as sk_clone

    # PCA projection
    X_all = np.array(X_train)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Clone and retrain on 2-D
    clf_2d = sk_clone(estimator)
    clf_2d.fit(X_2d, np.array(y_train))

    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Decision surface
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z.astype(float),
        colorscale="RdBu",
        opacity=0.35,
        showscale=False,
        contours=dict(showlines=False),
        name="boundary",
    ))

    # Training points
    y_arr = np.array(y_train)
    for cls in np.unique(y_arr):
        mask = y_arr == cls
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1],
            mode="markers",
            marker=dict(size=5, opacity=0.6),
            name=f"Train – class {cls}",
        ))

    # Test points (if provided)
    if X_test is not None and y_test is not None:
        X_test_2d = pca.transform(np.array(X_test))
        yt = np.array(y_test)
        for cls in np.unique(yt):
            mask = yt == cls
            fig.add_trace(go.Scatter(
                x=X_test_2d[mask, 0], y=X_test_2d[mask, 1],
                mode="markers",
                marker=dict(size=7, symbol="x", opacity=0.8),
                name=f"Test – class {cls}",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
        height=520,
    )
    return fig
