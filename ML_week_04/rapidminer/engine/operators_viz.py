"""
operators_viz.py – Visualization operators.

Implements: Data Distribution, Scatter Plot, Box Plot, Correlation Heatmap,
ROC Curve, Parallel Coordinates.

Each operator produces a matplotlib Figure stored in the output port so the
results panel can embed it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # headless by default; overridden when embedded in tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from engine.operator_base import (
    Operator,
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
    PortType,
    register_operator,
)

logger = logging.getLogger(__name__)


def _label_col(df: pd.DataFrame) -> str:
    roles = df.attrs.get("_roles", {})
    for col, role in roles.items():
        if role == "label" and col in df.columns:
            return col
    return df.columns[-1]


# ═══════════════════════════════════════════════════════════════════════════
# Data Distribution
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class DataDistribution(Operator):
    op_type = "Data Distribution"
    category = OpCategory.VISUALIZATION
    description = "Histogram + KDE for a selected column."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("column", ParamKind.COLUMN, default="", description="Column to plot."),
            ParamSpec("bins", ParamKind.INT, default=30, min_val=5, max_val=200, description="Number of bins."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        col = self.get_param("column") or df.select_dtypes(include="number").columns[0]
        bins = int(self.get_param("bins") or 30)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color="#7c3aed")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        return {"figure": fig}


# ═══════════════════════════════════════════════════════════════════════════
# Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ScatterPlotOp(Operator):
    op_type = "Scatter Plot"
    category = OpCategory.VISUALIZATION
    description = "2‑D scatter plot with optional colour and size."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("x", ParamKind.COLUMN, default="", description="X‑axis column."),
            ParamSpec("y", ParamKind.COLUMN, default="", description="Y‑axis column."),
            ParamSpec("color_by", ParamKind.COLUMN, default="", description="Colour‑by column (optional)."),
            ParamSpec("size_by", ParamKind.COLUMN, default="", description="Size‑by column (optional)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        x = self.get_param("x") or df.columns[0]
        y = self.get_param("y") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        color_by = self.get_param("color_by") or None
        size_by = self.get_param("size_by") or None

        fig, ax = plt.subplots(figsize=(7, 5))
        kwargs: Dict[str, Any] = {}
        if color_by and color_by in df.columns:
            kwargs["hue"] = color_by
        if size_by and size_by in df.columns:
            kwargs["size"] = size_by
        sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
        ax.set_title(f"Scatter: {x} vs {y}")
        fig.tight_layout()
        return {"figure": fig}


# ═══════════════════════════════════════════════════════════════════════════
# Box Plot
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class BoxPlotOp(Operator):
    op_type = "Box Plot"
    category = OpCategory.VISUALIZATION
    description = "Box plot of a numeric column grouped by a categorical column."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("x", ParamKind.COLUMN, default="", description="Categorical axis."),
            ParamSpec("y", ParamKind.COLUMN, default="", description="Numeric axis."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        x = self.get_param("x") or df.select_dtypes(include="object").columns[0] if len(df.select_dtypes(include="object").columns) else df.columns[0]
        y = self.get_param("y") or df.select_dtypes(include="number").columns[0]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x=x, y=y, ax=ax, palette="Set2")
        ax.set_title(f"Box Plot: {y} by {x}")
        fig.tight_layout()
        return {"figure": fig}


# ═══════════════════════════════════════════════════════════════════════════
# Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class CorrelationHeatmap(Operator):
    op_type = "Correlation Heatmap"
    category = OpCategory.VISUALIZATION
    description = "Heatmap of the correlation matrix."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        num = df.select_dtypes(include="number")
        corr = num.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        return {"figure": fig}


# ═══════════════════════════════════════════════════════════════════════════
# ROC Curve
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ROCCurveOp(Operator):
    op_type = "ROC Curve"
    category = OpCategory.VISUALIZATION
    description = "Plot the Receiver Operating Characteristic curve."

    def _build_ports(self) -> None:
        self.inputs["performance"] = Port("performance", PortType.PERFORMANCE)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.metrics import roc_curve, auc
        perf: Dict[str, Any] = inputs["performance"]
        fig, ax = plt.subplots(figsize=(6, 5))

        if "y_proba" in perf and perf.get("y_true"):
            y_true = np.array(perf["y_true"])
            y_proba = np.array(perf["y_proba"])
            classes = perf.get("labels", sorted(set(y_true)))
            if len(classes) == 2 and y_proba.ndim == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color="#7c3aed", lw=2, label=f"AUC = {roc_auc:.3f}")
            elif y_proba.ndim == 2:
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(y_true, classes=classes)
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(yb[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f"Class {cls} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return {"figure": fig}


# ═══════════════════════════════════════════════════════════════════════════
# Parallel Coordinates
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ParallelCoordinates(Operator):
    op_type = "Parallel Coordinates"
    category = OpCategory.VISUALIZATION
    description = "Parallel coordinates plot coloured by label / cluster."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["figure"] = Port("figure", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("color_column", ParamKind.COLUMN, default="", description="Column used for colour (label / cluster)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from pandas.plotting import parallel_coordinates
        df: pd.DataFrame = inputs["in"]
        color_col = self.get_param("color_column")
        if not color_col or color_col not in df.columns:
            color_col = _label_col(df) if _label_col(df) in df.columns else df.columns[-1]

        # Keep only numeric + class column
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if color_col not in num_cols:
            plot_cols = num_cols + [color_col]
        else:
            plot_cols = num_cols
        sub = df[plot_cols].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        parallel_coordinates(sub, color_col, ax=ax, colormap="viridis")
        ax.set_title("Parallel Coordinates")
        fig.tight_layout()
        return {"figure": fig}
