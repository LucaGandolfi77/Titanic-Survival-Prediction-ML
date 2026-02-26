"""
operators_eval.py – Evaluation operators.

Implements: Apply Model, Performance (Classification), Performance (Regression),
Performance (Clustering), Cross Validation, Feature Importance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

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


def _numeric_X(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return df.drop(columns=[label, "prediction"], errors="ignore").select_dtypes(include="number").fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
# Apply Model
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ApplyModel(Operator):
    op_type = "Apply Model"
    category = OpCategory.EVALUATION
    description = "Apply a trained model to an ExampleSet to produce predictions."

    def _build_ports(self) -> None:
        self.inputs["model"] = Port("model", PortType.MODEL)
        self.inputs["unlabelled"] = Port("unlabelled", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model = inputs["model"]
        df: pd.DataFrame = inputs["unlabelled"].copy()
        label = _label_col(df)
        X = _numeric_X(df, label)

        # Use the same feature set the model was trained on if possible
        if hasattr(model, "feature_names_in_"):
            feats = [f for f in model.feature_names_in_ if f in X.columns]
            missing = [f for f in model.feature_names_in_ if f not in X.columns]
            for m in missing:
                X[m] = 0
            X = X[list(model.feature_names_in_)]

        preds = model.predict(X)
        df["prediction"] = preds

        # Probabilities for classification
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                for i, cls_label in enumerate(model.classes_):
                    df[f"confidence_{cls_label}"] = proba[:, i]
            except Exception:
                pass

        df.attrs = inputs["unlabelled"].attrs.copy()
        logger.info("Applied model – %d predictions", len(preds))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Classification)
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class PerformanceClassification(Operator):
    op_type = "Performance (Classification)"
    category = OpCategory.EVALUATION
    description = "Classification metrics: accuracy, precision, recall, F1, AUC, confusion matrix."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["performance"] = Port("performance", PortType.PERFORMANCE)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score,
        )
        df: pd.DataFrame = inputs["in"]
        label = _label_col(df)
        if "prediction" not in df.columns:
            raise ValueError("Input must have a 'prediction' column (use Apply Model first).")

        y_true = df[label]
        y_pred = df["prediction"]
        average = "binary" if y_true.nunique() <= 2 else "weighted"

        perf: Dict[str, Any] = {
            "type": "classification",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "labels": sorted(y_true.unique().tolist()),
        }

        # ROC‑AUC (only if probabilities exist)
        proba_cols = [c for c in df.columns if c.startswith("confidence_")]
        if proba_cols:
            try:
                if y_true.nunique() == 2:
                    perf["roc_auc"] = roc_auc_score(y_true, df[proba_cols[-1]])
                else:
                    from sklearn.preprocessing import label_binarize
                    yb = label_binarize(y_true, classes=perf["labels"])
                    proba_arr = df[sorted(proba_cols)].values
                    perf["roc_auc"] = roc_auc_score(yb, proba_arr, multi_class="ovr", average="weighted")
            except Exception:
                perf["roc_auc"] = None
        else:
            perf["roc_auc"] = None

        # Store true/pred arrays for ROC plots
        perf["y_true"] = y_true.tolist()
        perf["y_pred"] = y_pred.tolist()
        if proba_cols:
            perf["y_proba"] = df[sorted(proba_cols)].values.tolist()

        logger.info("Classification performance: acc=%.4f  f1=%.4f", perf["accuracy"], perf["f1"])
        return {"performance": perf, "out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Regression)
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class PerformanceRegression(Operator):
    op_type = "Performance (Regression)"
    category = OpCategory.EVALUATION
    description = "Regression metrics: MAE, MSE, RMSE, R²."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["performance"] = Port("performance", PortType.PERFORMANCE)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
        )
        df: pd.DataFrame = inputs["in"]
        label = _label_col(df)
        if "prediction" not in df.columns:
            raise ValueError("Input must have a 'prediction' column.")

        y_true = df[label].astype(float)
        y_pred = df["prediction"].astype(float)
        mse = mean_squared_error(y_true, y_pred)

        perf: Dict[str, Any] = {
            "type": "regression",
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_true, y_pred),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }
        logger.info("Regression performance: R²=%.4f  RMSE=%.4f", perf["r2"], perf["rmse"])
        return {"performance": perf, "out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Performance (Clustering)
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class PerformanceClustering(Operator):
    op_type = "Performance (Clustering)"
    category = OpCategory.EVALUATION
    description = "Clustering metrics: silhouette score, inertia, cluster distribution."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.inputs["model"] = Port("model", PortType.MODEL, optional=True)
        self.outputs["performance"] = Port("performance", PortType.PERFORMANCE)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.metrics import silhouette_score
        df: pd.DataFrame = inputs["in"]
        model = inputs.get("model")

        if "cluster" not in df.columns:
            raise ValueError("Input must have a 'cluster' column.")

        labels = df["cluster"]
        num = df.select_dtypes(include="number").drop(columns=["cluster"], errors="ignore").fillna(0)
        n_labels = labels.nunique()

        perf: Dict[str, Any] = {"type": "clustering"}
        if n_labels > 1 and n_labels < len(labels):
            perf["silhouette"] = silhouette_score(num, labels)
        else:
            perf["silhouette"] = None

        if model and hasattr(model, "inertia_"):
            perf["inertia"] = float(model.inertia_)
        else:
            perf["inertia"] = None

        perf["cluster_distribution"] = labels.value_counts().to_dict()
        logger.info("Clustering performance: silhouette=%s", perf["silhouette"])
        return {"performance": perf, "out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Cross Validation
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class CrossValidation(Operator):
    op_type = "Cross Validation"
    category = OpCategory.EVALUATION
    description = "K‑fold cross‑validation (supports classification & regression)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["performance"] = Port("performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("k", ParamKind.INT, default=5, min_val=2, max_val=50, description="Number of folds."),
            ParamSpec("stratified", ParamKind.BOOL, default=True, description="Stratified folds."),
            ParamSpec("shuffle", ParamKind.BOOL, default=True, description="Shuffle before splitting."),
            ParamSpec("seed", ParamKind.INT, default=42, description="Random seed."),
            ParamSpec("estimator", ParamKind.CHOICE, default="Random Forest",
                      choices=["Logistic Regression", "Decision Tree", "Random Forest",
                               "Gradient Boosting", "SVM", "KNN", "Naive Bayes",
                               "Linear Regression", "Ridge", "Lasso"],
                      description="Estimator to use inside each fold."),
            ParamSpec("scoring", ParamKind.CHOICE, default="accuracy",
                      choices=["accuracy", "f1_weighted", "r2", "neg_mean_squared_error"],
                      description="Scoring metric."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from engine.operator_base import get_operator_class

        df: pd.DataFrame = inputs["in"]
        label = _label_col(df)
        X = df.drop(columns=[label], errors="ignore").select_dtypes(include="number").fillna(0)
        y = df[label]

        k = int(self.get_param("k") or 5)
        seed = int(self.get_param("seed") or 42)
        shuffle = bool(self.get_param("shuffle"))
        scoring = self.get_param("scoring") or "accuracy"

        # Build scikit‑learn estimator from op_type
        est_name = self.get_param("estimator") or "Random Forest"
        est_cls = get_operator_class(est_name)
        temp_op = est_cls()
        temp_op.params = temp_op.params.copy()

        # Map op → sklearn class
        sklearn_map = {
            "Logistic Regression": ("sklearn.linear_model", "LogisticRegression"),
            "Decision Tree": ("sklearn.tree", "DecisionTreeClassifier"),
            "Random Forest": ("sklearn.ensemble", "RandomForestClassifier"),
            "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingClassifier"),
            "SVM": ("sklearn.svm", "SVC"),
            "KNN": ("sklearn.neighbors", "KNeighborsClassifier"),
            "Naive Bayes": ("sklearn.naive_bayes", "GaussianNB"),
            "Linear Regression": ("sklearn.linear_model", "LinearRegression"),
            "Ridge": ("sklearn.linear_model", "Ridge"),
            "Lasso": ("sklearn.linear_model", "Lasso"),
        }
        mod_name, cls_name = sklearn_map.get(est_name, ("sklearn.ensemble", "RandomForestClassifier"))
        import importlib
        mod = importlib.import_module(mod_name)
        sk_cls = getattr(mod, cls_name)
        estimator = sk_cls()

        if self.get_param("stratified") and scoring in ("accuracy", "f1_weighted"):
            cv = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        else:
            cv = KFold(n_splits=k, shuffle=shuffle, random_state=seed)

        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        perf = {
            "type": "cross_validation",
            "scoring": scoring,
            "k": k,
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "estimator": est_name,
        }
        logger.info("CV(%s, k=%d): mean=%.4f  std=%.4f", est_name, k, perf["mean"], perf["std"])
        return {"performance": perf}


# ═══════════════════════════════════════════════════════════════════════════
# Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class FeatureImportance(Operator):
    op_type = "Feature Importance"
    category = OpCategory.EVALUATION
    description = "Extract and rank feature importances from a trained model."

    def _build_ports(self) -> None:
        self.inputs["model"] = Port("model", PortType.MODEL)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model = inputs["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(importances))]
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            importances = coef
            names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(importances))]
        else:
            return {"out": pd.DataFrame({"feature": ["N/A"], "importance": [0.0]})}

        result = pd.DataFrame({"feature": names, "importance": importances})
        result = result.sort_values("importance", ascending=False).reset_index(drop=True)
        return {"out": result}
