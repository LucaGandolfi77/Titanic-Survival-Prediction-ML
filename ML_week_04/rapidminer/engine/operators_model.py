"""
operators_model.py – Machine‑learning model operators.

Classification: Logistic Regression, Decision Tree, Random Forest,
                Gradient Boosting, SVM, KNN, Naive Bayes.
Regression:     Linear Regression, Ridge, Lasso.
Clustering:     KMeans, DBSCAN, Agglomerative.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

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


def _prepare_Xy(df: pd.DataFrame, label: str):
    """Return X (numeric, NaN→0) and y from *df*."""
    y = df[label]
    X = df.drop(columns=[label], errors="ignore").select_dtypes(include="number").fillna(0)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class LogisticRegressionOp(Operator):
    op_type = "Logistic Regression"
    category = OpCategory.MODEL
    description = "Train a Logistic Regression classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("C", ParamKind.FLOAT, default=1.0, min_val=0.001, description="Inverse regularization strength."),
            ParamSpec("max_iter", ParamKind.INT, default=200, min_val=10, description="Max iterations."),
            ParamSpec("solver", ParamKind.CHOICE, default="lbfgs", choices=["lbfgs", "liblinear", "saga"], description="Solver."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        df: pd.DataFrame = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = LogisticRegression(
            C=float(self.get_param("C") or 1),
            max_iter=int(self.get_param("max_iter") or 200),
            solver=self.get_param("solver") or "lbfgs",
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        logger.info("Logistic Regression: train acc=%.4f", acc)
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class DecisionTreeOp(Operator):
    op_type = "Decision Tree"
    category = OpCategory.MODEL
    description = "Train a Decision Tree classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("max_depth", ParamKind.INT, default=5, min_val=1, max_val=100, description="Maximum depth."),
            ParamSpec("min_samples_split", ParamKind.INT, default=2, min_val=2, description="Min samples to split."),
            ParamSpec("criterion", ParamKind.CHOICE, default="gini", choices=["gini", "entropy", "log_loss"], description="Split criterion."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = DecisionTreeClassifier(
            max_depth=int(self.get_param("max_depth") or 5),
            min_samples_split=int(self.get_param("min_samples_split") or 2),
            criterion=self.get_param("criterion") or "gini",
            random_state=42,
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class RandomForestOp(Operator):
    op_type = "Random Forest"
    category = OpCategory.MODEL
    description = "Train a Random Forest classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_estimators", ParamKind.INT, default=100, min_val=10, max_val=2000, description="Number of trees."),
            ParamSpec("max_depth", ParamKind.INT, default=10, min_val=1, max_val=100, description="Maximum depth (0=None)."),
            ParamSpec("max_features", ParamKind.CHOICE, default="sqrt", choices=["sqrt", "log2", "all"], description="Features per split."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        md = int(self.get_param("max_depth") or 10)
        mf = self.get_param("max_features")
        if mf == "all":
            mf = None
        est = RandomForestClassifier(
            n_estimators=int(self.get_param("n_estimators") or 100),
            max_depth=md if md > 0 else None,
            max_features=mf,
            random_state=42, n_jobs=-1,
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class GradientBoostingOp(Operator):
    op_type = "Gradient Boosting"
    category = OpCategory.MODEL
    description = "Train a Gradient Boosting classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_estimators", ParamKind.INT, default=100, min_val=10, max_val=2000, description="Number of boosting stages."),
            ParamSpec("learning_rate", ParamKind.FLOAT, default=0.1, min_val=0.001, max_val=1.0, description="Learning rate."),
            ParamSpec("max_depth", ParamKind.INT, default=3, min_val=1, max_val=30, description="Max depth per tree."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = GradientBoostingClassifier(
            n_estimators=int(self.get_param("n_estimators") or 100),
            learning_rate=float(self.get_param("learning_rate") or 0.1),
            max_depth=int(self.get_param("max_depth") or 3),
            random_state=42,
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class SVMOp(Operator):
    op_type = "SVM"
    category = OpCategory.MODEL
    description = "Train a Support Vector Machine classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("C", ParamKind.FLOAT, default=1.0, min_val=0.001, description="Regularization parameter."),
            ParamSpec("kernel", ParamKind.CHOICE, default="rbf", choices=["rbf", "linear", "poly", "sigmoid"], description="Kernel type."),
            ParamSpec("gamma", ParamKind.CHOICE, default="scale", choices=["scale", "auto"], description="Kernel coefficient."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = SVC(
            C=float(self.get_param("C") or 1),
            kernel=self.get_param("kernel") or "rbf",
            gamma=self.get_param("gamma") or "scale",
            probability=True, random_state=42,
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class KNNOp(Operator):
    op_type = "KNN"
    category = OpCategory.MODEL
    description = "K‑Nearest Neighbours classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_neighbors", ParamKind.INT, default=5, min_val=1, max_val=200, description="Number of neighbours."),
            ParamSpec("metric", ParamKind.CHOICE, default="minkowski", choices=["minkowski", "euclidean", "manhattan", "cosine"], description="Distance metric."),
            ParamSpec("weights", ParamKind.CHOICE, default="uniform", choices=["uniform", "distance"], description="Weight function."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = KNeighborsClassifier(
            n_neighbors=int(self.get_param("n_neighbors") or 5),
            metric=self.get_param("metric") or "minkowski",
            weights=self.get_param("weights") or "uniform",
        )
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


@register_operator
class NaiveBayesOp(Operator):
    op_type = "Naive Bayes"
    category = OpCategory.MODEL
    description = "Gaussian Naive Bayes classifier."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("var_smoothing", ParamKind.FLOAT, default=1e-9, min_val=0.0, description="Variance smoothing."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = GaussianNB(var_smoothing=float(self.get_param("var_smoothing") or 1e-9))
        est.fit(X, y)
        acc = accuracy_score(y, est.predict(X))
        return {"model": est, "training_performance": {"accuracy_train": acc, "model_type": self.op_type}}


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class LinearRegressionOp(Operator):
    op_type = "Linear Regression"
    category = OpCategory.MODEL
    description = "Ordinary Least‑Squares linear regression."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("fit_intercept", ParamKind.BOOL, default=True, description="Fit intercept."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = LinearRegression(fit_intercept=bool(self.get_param("fit_intercept")))
        est.fit(X, y)
        r2 = r2_score(y, est.predict(X))
        return {"model": est, "training_performance": {"r2_train": r2, "model_type": self.op_type}}


@register_operator
class RidgeOp(Operator):
    op_type = "Ridge"
    category = OpCategory.MODEL
    description = "Ridge (L2‑penalised) regression."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("alpha", ParamKind.FLOAT, default=1.0, min_val=0.0, description="Regularization strength."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = Ridge(alpha=float(self.get_param("alpha") or 1.0))
        est.fit(X, y)
        r2 = r2_score(y, est.predict(X))
        return {"model": est, "training_performance": {"r2_train": r2, "model_type": self.op_type}}


@register_operator
class LassoOp(Operator):
    op_type = "Lasso"
    category = OpCategory.MODEL
    description = "Lasso (L1‑penalised) regression."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["training_performance"] = Port("training_performance", PortType.PERFORMANCE)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("alpha", ParamKind.FLOAT, default=1.0, min_val=0.0, description="Regularization strength."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.linear_model import Lasso
        from sklearn.metrics import r2_score
        df = inputs["training"]
        label = _label_col(df)
        X, y = _prepare_Xy(df, label)
        est = Lasso(alpha=float(self.get_param("alpha") or 1.0), max_iter=5000)
        est.fit(X, y)
        r2 = r2_score(y, est.predict(X))
        return {"model": est, "training_performance": {"r2_train": r2, "model_type": self.op_type}}


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class KMeansOp(Operator):
    op_type = "KMeans"
    category = OpCategory.MODEL
    description = "K‑Means clustering."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_clusters", ParamKind.INT, default=3, min_val=2, max_val=100, description="Number of clusters."),
            ParamSpec("init", ParamKind.CHOICE, default="k-means++", choices=["k-means++", "random"], description="Initialization method."),
            ParamSpec("max_iter", ParamKind.INT, default=300, min_val=10, description="Max iterations."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.cluster import KMeans
        df: pd.DataFrame = inputs["training"].copy()
        num = df.select_dtypes(include="number").fillna(0)
        est = KMeans(
            n_clusters=int(self.get_param("n_clusters") or 3),
            init=self.get_param("init") or "k-means++",
            max_iter=int(self.get_param("max_iter") or 300),
            random_state=42, n_init=10,
        )
        labels = est.fit_predict(num)
        df["cluster"] = labels
        return {"model": est, "out": df}


@register_operator
class DBSCANOp(Operator):
    op_type = "DBSCAN"
    category = OpCategory.MODEL
    description = "Density‑Based Spatial Clustering of Applications with Noise."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("eps", ParamKind.FLOAT, default=0.5, min_val=0.001, description="Maximum distance between samples."),
            ParamSpec("min_samples", ParamKind.INT, default=5, min_val=1, description="Minimum samples in a core neighbourhood."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.cluster import DBSCAN
        df: pd.DataFrame = inputs["training"].copy()
        num = df.select_dtypes(include="number").fillna(0)
        est = DBSCAN(
            eps=float(self.get_param("eps") or 0.5),
            min_samples=int(self.get_param("min_samples") or 5),
        )
        labels = est.fit_predict(num)
        df["cluster"] = labels
        return {"model": est, "out": df}


@register_operator
class AgglomerativeOp(Operator):
    op_type = "Agglomerative"
    category = OpCategory.MODEL
    description = "Agglomerative (hierarchical) clustering."

    def _build_ports(self) -> None:
        self.inputs["training"] = Port("training", PortType.EXAMPLE_SET)
        self.outputs["model"] = Port("model", PortType.MODEL)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("n_clusters", ParamKind.INT, default=3, min_val=2, max_val=100, description="Number of clusters."),
            ParamSpec("linkage", ParamKind.CHOICE, default="ward", choices=["ward", "complete", "average", "single"], description="Linkage criterion."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from sklearn.cluster import AgglomerativeClustering
        df: pd.DataFrame = inputs["training"].copy()
        num = df.select_dtypes(include="number").fillna(0)
        est = AgglomerativeClustering(
            n_clusters=int(self.get_param("n_clusters") or 3),
            linkage=self.get_param("linkage") or "ward",
        )
        labels = est.fit_predict(num)
        df["cluster"] = labels
        return {"model": est, "out": df}
