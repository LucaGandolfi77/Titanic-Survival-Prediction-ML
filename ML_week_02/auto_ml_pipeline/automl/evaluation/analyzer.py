"""
Model analysis – feature importance, learning curves, SHAP (optional).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore


class ModelAnalyzer:
    """Extract and visualise feature importance, learning curves, etc."""

    def __init__(self, config) -> None:
        self.shap_enabled: bool = getattr(config, "shap_enabled", False)
        self.top_n: int = getattr(config, "top_n_features", 20)

    def feature_importance(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Extract feature importance from tree-based or linear models."""
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                imp = np.abs(coef).mean(axis=0)
            else:
                imp = np.abs(coef)
        # Stacking / Voting wrappers
        elif hasattr(model, "estimators_"):
            for est_name, est in (
                model.named_estimators_.items()
                if hasattr(model, "named_estimators_")
                else [(str(i), e) for i, e in enumerate(model.estimators_)]
            ):
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    break

        if imp is None:
            logger.warning("Could not extract feature importances.")
            return pd.DataFrame({"feature": feature_names, "importance": 0.0})

        # Align lengths (safety)
        n = min(len(imp), len(feature_names))
        df = pd.DataFrame({
            "feature": feature_names[:n],
            "importance": imp[:n],
        })
        df = df.sort_values("importance", ascending=False).head(self.top_n)
        return df.reset_index(drop=True)

    def importance_chart(self, imp_df: pd.DataFrame) -> Optional[Any]:
        """Return a Plotly bar chart of feature importances."""
        if go is None:
            return None
        df = imp_df.sort_values("importance", ascending=True).tail(self.top_n)
        fig = go.Figure(
            go.Bar(
                x=df["importance"],
                y=df["feature"],
                orientation="h",
                marker_color="#636efa",
            )
        )
        fig.update_layout(
            title="Feature Importances (Top-{})".format(len(df)),
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(df) * 25),
            margin=dict(l=200),
        )
        return fig

    def shap_summary(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Optional[Any]:
        """Compute SHAP values (if shap is installed and enabled)."""
        if not self.shap_enabled:
            return None
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:500])
            return {"shap_values": shap_values, "feature_names": feature_names}
        except Exception as exc:
            logger.warning(f"SHAP analysis failed: {exc}")
            return None
