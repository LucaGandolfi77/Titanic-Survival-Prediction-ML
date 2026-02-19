"""
explainer.py – Feature importance & SHAP explanations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def feature_importance_builtin(
    estimator: BaseEstimator,
    feature_names: list[str],
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    """Extract built-in feature_importances_ (tree-based models)."""
    if not hasattr(estimator, "feature_importances_"):
        return None
    imp = estimator.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def feature_importance_coefficients(
    estimator: BaseEstimator,
    feature_names: list[str],
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    """Extract linear-model coefficients as importance."""
    if not hasattr(estimator, "coef_"):
        return None
    coefs = np.abs(estimator.coef_).flatten()
    if len(coefs) != len(feature_names):
        # Multi-class: average across classes
        coefs = np.abs(estimator.coef_).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": coefs,
    }).sort_values("importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def permutation_feature_importance(
    estimator: BaseEstimator,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    top_n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Model-agnostic permutation importance."""
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def get_feature_importance(
    estimator: BaseEstimator,
    feature_names: list[str],
    X_test: pd.DataFrame | np.ndarray = None,
    y_test: pd.Series | np.ndarray = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Smart dispatcher – tries built-in → coefficients → permutation."""
    # Try built-in
    df = feature_importance_builtin(estimator, feature_names, top_n)
    if df is not None:
        return df

    # Try coefficients
    df = feature_importance_coefficients(estimator, feature_names, top_n)
    if df is not None:
        return df

    # Fallback to permutation
    if X_test is not None and y_test is not None:
        return permutation_feature_importance(
            estimator, X_test, y_test, feature_names, top_n=top_n
        )

    return pd.DataFrame(columns=["feature", "importance"])


def plot_feature_importance(imp_df: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart of feature importance."""
    if imp_df.empty:
        return go.Figure()
    col = "importance" if "importance" in imp_df.columns else "importance_mean"
    df = imp_df.sort_values(col, ascending=True).tail(20)
    fig = px.bar(
        df, x=col, y="feature",
        orientation="h",
        title=title,
        template="plotly_white",
        color=col,
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=max(350, 25 * len(df)), yaxis_title="", coloraxis_showscale=False)
    return fig


def shap_summary(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    max_display: int = 20,
) -> Optional[go.Figure]:
    """SHAP summary plot (bar) using the shap library.

    Returns None if SHAP cannot explain this model.
    """
    try:
        import shap

        # Pick the right explainer automatically
        try:
            explainer = shap.TreeExplainer(estimator)
        except Exception:
            # Fallback to Kernel/Linear
            if hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, X)
            else:
                # KernelExplainer is slow – sample
                sample = X[:100] if len(X) > 100 else X
                explainer = shap.KernelExplainer(estimator.predict, sample)

        shap_values = explainer.shap_values(X[:200] if len(X) > 200 else X)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_names = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]

        df = pd.DataFrame({
            "feature": feature_names,
            "mean_|SHAP|": mean_abs,
        }).sort_values("mean_|SHAP|", ascending=True).tail(max_display)

        fig = px.bar(
            df, x="mean_|SHAP|", y="feature",
            orientation="h",
            title="SHAP Feature Importance",
            template="plotly_white",
            color="mean_|SHAP|",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            height=max(350, 25 * len(df)),
            yaxis_title="",
            coloraxis_showscale=False,
        )
        return fig

    except Exception:
        return None
