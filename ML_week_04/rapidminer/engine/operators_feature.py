"""
operators_feature.py – Feature‑engineering operators.

Implements: PCA, Variance Threshold, Forward Selection, Backward Elimination,
Correlation Matrix, Weight by Correlation, One Hot Encoding, Label Encoding,
Target Encoding.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SkPCA
from sklearn.feature_selection import VarianceThreshold as SkVarThresh

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


# ── helpers ────────────────────────────────────────────────────────────────

def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns, NaN filled with 0."""
    return df.select_dtypes(include="number").fillna(0)


def _label_col(df: pd.DataFrame) -> str:
    """Get the label column from attrs metadata, or last column."""
    roles = df.attrs.get("_roles", {})
    for col, role in roles.items():
        if role == "label" and col in df.columns:
            return col
    return df.columns[-1]


# ═══════════════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class PCAOp(Operator):
    op_type = "PCA"
    category = OpCategory.FEATURE
    description = "Principal Component Analysis – reduce dimensionality."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_components", ParamKind.INT, default=2, min_val=1, max_val=500,
                      description="Number of principal components."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        n = int(self.get_param("n_components") or 2)
        num = _numeric_df(df)
        n = min(n, num.shape[1], num.shape[0])
        pca = SkPCA(n_components=n)
        transformed = pca.fit_transform(num)
        new_cols = [f"PC{i+1}" for i in range(n)]
        pca_df = pd.DataFrame(transformed, columns=new_cols, index=df.index)
        # Keep non‑numeric columns
        non_num = df.select_dtypes(exclude="number")
        result = pd.concat([non_num.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
        result.attrs = df.attrs.copy()
        return {"out": result, "model": pca}


# ═══════════════════════════════════════════════════════════════════════════
# Feature Selection – Variance Threshold
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class VarianceThresholdOp(Operator):
    op_type = "Variance Threshold"
    category = OpCategory.FEATURE
    description = "Remove features with variance below a threshold."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("threshold", ParamKind.FLOAT, default=0.0, min_val=0.0,
                      description="Minimum variance to keep a feature."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        threshold = float(self.get_param("threshold") or 0.0)
        num = _numeric_df(df)
        sel = SkVarThresh(threshold=threshold)
        sel.fit(num)
        keep = num.columns[sel.get_support()].tolist()
        non_num_cols = df.select_dtypes(exclude="number").columns.tolist()
        result = df[non_num_cols + keep].copy()
        result.attrs = df.attrs.copy()
        return {"out": result}


# ═══════════════════════════════════════════════════════════════════════════
# Forward Selection
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ForwardSelection(Operator):
    op_type = "Forward Selection"
    category = OpCategory.FEATURE
    description = "Greedy forward feature selection based on cross‑val score."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("max_features", ParamKind.INT, default=5, min_val=1, max_val=200,
                      description="Max features to select."),
            ParamSpec("scoring", ParamKind.CHOICE, default="accuracy",
                      choices=["accuracy", "f1_macro", "r2", "neg_mean_squared_error"],
                      description="Scoring metric."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        df: pd.DataFrame = inputs["in"].copy()
        label = _label_col(df)
        y = df[label]
        X = _numeric_df(df.drop(columns=[label], errors="ignore"))
        if X.empty:
            return {"out": df}

        max_f = min(int(self.get_param("max_features") or 5), X.shape[1])
        scoring = self.get_param("scoring") or "accuracy"
        selected: List[str] = []
        remaining = list(X.columns)
        est = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        for _ in range(max_f):
            best_score = -np.inf
            best_col = None
            for col in remaining:
                trial = selected + [col]
                score = cross_val_score(est, X[trial], y, cv=3, scoring=scoring).mean()
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col is None:
                break
            selected.append(best_col)
            remaining.remove(best_col)
            logger.info("Forward sel: added '%s' (score=%.4f)", best_col, best_score)

        non_num = df.select_dtypes(exclude="number").columns.tolist()
        keep = non_num + selected + ([label] if label not in selected and label not in non_num else [])
        result = df[[c for c in keep if c in df.columns]].copy()
        result.attrs = df.attrs.copy()
        return {"out": result}


# ═══════════════════════════════════════════════════════════════════════════
# Backward Elimination
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class BackwardElimination(Operator):
    op_type = "Backward Elimination"
    category = OpCategory.FEATURE
    description = "Greedy backward feature elimination based on cross‑val score."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("scoring", ParamKind.CHOICE, default="accuracy",
                      choices=["accuracy", "f1_macro", "r2", "neg_mean_squared_error"],
                      description="Scoring metric."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        df: pd.DataFrame = inputs["in"].copy()
        label = _label_col(df)
        y = df[label]
        X = _numeric_df(df.drop(columns=[label], errors="ignore"))
        if X.shape[1] <= 1:
            return {"out": df}

        scoring = self.get_param("scoring") or "accuracy"
        est = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        remaining = list(X.columns)

        while len(remaining) > 1:
            scores = {}
            for col in remaining:
                trial = [c for c in remaining if c != col]
                scores[col] = cross_val_score(est, X[trial], y, cv=3, scoring=scoring).mean()
            worst = min(scores, key=scores.get)  # type: ignore[arg-type]
            base_score = cross_val_score(est, X[remaining], y, cv=3, scoring=scoring).mean()
            if scores[worst] >= base_score:
                remaining.remove(worst)
                logger.info("Backward elim: removed '%s'", worst)
            else:
                break

        non_num = df.select_dtypes(exclude="number").columns.tolist()
        keep = non_num + remaining + ([label] if label not in remaining and label not in non_num else [])
        result = df[[c for c in keep if c in df.columns]].copy()
        result.attrs = df.attrs.copy()
        return {"out": result}


# ═══════════════════════════════════════════════════════════════════════════
# Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class CorrelationMatrix(Operator):
    op_type = "Correlation Matrix"
    category = OpCategory.FEATURE
    description = "Compute the correlation matrix and output as ExampleSet."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("method", ParamKind.CHOICE, default="pearson",
                      choices=["pearson", "spearman", "kendall"],
                      description="Correlation method."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        method = self.get_param("method") or "pearson"
        corr = _numeric_df(df).corr(method=method)
        corr.insert(0, "feature", corr.index)
        return {"out": corr.reset_index(drop=True)}


# ═══════════════════════════════════════════════════════════════════════════
# Weight by Correlation
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class WeightByCorrelation(Operator):
    op_type = "Weight by Correlation"
    category = OpCategory.FEATURE
    description = "Rank features by absolute correlation with label."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        label = _label_col(df)
        num = _numeric_df(df)
        if label not in num.columns:
            return {"out": pd.DataFrame({"feature": [], "weight": []})}
        corrs = num.corrwith(num[label]).drop(label, errors="ignore").abs().sort_values(ascending=False)
        result = pd.DataFrame({"feature": corrs.index, "weight": corrs.values})
        return {"out": result}


# ═══════════════════════════════════════════════════════════════════════════
# One Hot Encoding
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class OneHotEncodingOp(Operator):
    op_type = "One Hot Encoding"
    category = OpCategory.FEATURE
    description = "One‑hot encode selected categorical columns."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Columns to encode (empty = all object)."),
            ParamSpec("drop_first", ParamKind.BOOL, default=False, description="Drop first dummy to avoid multicollinearity."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        cols = self._parse_cols(df)
        drop_first = bool(self.get_param("drop_first"))
        df = pd.get_dummies(df, columns=cols, drop_first=drop_first)
        # Ensure column names are strings
        df.columns = [str(c) for c in df.columns]
        return {"out": df}

    def _parse_cols(self, df: pd.DataFrame) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            cols = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list) and raw:
            cols = raw
        else:
            cols = df.select_dtypes(include="object").columns.tolist()
        return [c for c in cols if c in df.columns]


# ═══════════════════════════════════════════════════════════════════════════
# Label Encoding
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class LabelEncodingOp(Operator):
    op_type = "Label Encoding"
    category = OpCategory.FEATURE
    description = "Encode categorical columns as integer labels."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Columns to encode (empty = all object)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.preprocessing import LabelEncoder
        df: pd.DataFrame = inputs["in"].copy()
        cols = self._parse_cols(df)
        for c in cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
        return {"out": df}

    def _parse_cols(self, df: pd.DataFrame) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            cols = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list) and raw:
            cols = raw
        else:
            cols = df.select_dtypes(include="object").columns.tolist()
        return [c for c in cols if c in df.columns]


# ═══════════════════════════════════════════════════════════════════════════
# Target Encoding
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class TargetEncodingOp(Operator):
    op_type = "Target Encoding"
    category = OpCategory.FEATURE
    description = "Encode categorical columns as the mean of the target per category."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Columns to encode."),
            ParamSpec("smoothing", ParamKind.FLOAT, default=1.0, min_val=0.0, description="Smoothing factor."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        label = _label_col(df)
        if label not in df.columns or not pd.api.types.is_numeric_dtype(df[label]):
            return {"out": df}

        cols = self._parse_cols(df)
        smoothing = float(self.get_param("smoothing") or 1.0)
        global_mean = df[label].mean()

        for c in cols:
            means = df.groupby(c)[label].agg(["mean", "count"])
            smooth = (means["count"] * means["mean"] + smoothing * global_mean) / (means["count"] + smoothing)
            df[c] = df[c].map(smooth)
        return {"out": df}

    def _parse_cols(self, df: pd.DataFrame) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            cols = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list) and raw:
            cols = raw
        else:
            cols = df.select_dtypes(include="object").columns.tolist()
        return [c for c in cols if c in df.columns]
