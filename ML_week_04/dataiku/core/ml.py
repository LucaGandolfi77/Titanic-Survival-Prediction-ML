"""
MLLab – train, evaluate, and export scikit-learn models.

Supports classification, regression, and clustering with
automated metric computation (accuracy, F1, RMSE, silhouette, …).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Optional: XGBoost
# ---------------------------------------------------------------------------

_HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None  # type: ignore[misc,assignment]
    XGBRegressor = None   # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

CLASSIFICATION_ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "defaults": {"max_iter": 1000, "C": 1.0},
        "params": {
            "C": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 10000, "default": 1000},
        },
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "defaults": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
            "max_depth": {"type": "int_or_none", "min": 1, "max": 50, "default": "None"},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
        },
    },
    "SVM": {
        "class": SVC,
        "defaults": {"C": 1.0, "kernel": "rbf", "probability": True},
        "params": {
            "C": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
            "kernel": {"type": "choice", "choices": ["rbf", "linear", "poly"], "default": "rbf"},
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "defaults": {"n_neighbors": 5},
        "params": {
            "n_neighbors": {"type": "int", "min": 1, "max": 50, "default": 5},
        },
    },
}

if _HAS_XGBOOST:
    CLASSIFICATION_ALGORITHMS["XGBoost"] = {
        "class": XGBClassifier,
        "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                      "use_label_encoder": False, "eval_metric": "logloss"},
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
            "max_depth": {"type": "int", "min": 1, "max": 20, "default": 6},
            "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
        },
    }


REGRESSION_ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "class": LinearRegression,
        "defaults": {},
        "params": {},
    },
    "Ridge": {
        "class": Ridge,
        "defaults": {"alpha": 1.0},
        "params": {
            "alpha": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
        },
    },
    "Lasso": {
        "class": Lasso,
        "defaults": {"alpha": 1.0, "max_iter": 5000},
        "params": {
            "alpha": {"type": "float", "min": 0.001, "max": 100, "default": 1.0},
        },
    },
    "Random Forest Regressor": {
        "class": RandomForestRegressor,
        "defaults": {"n_estimators": 100, "random_state": 42},
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
            "max_depth": {"type": "int_or_none", "min": 1, "max": 50, "default": "None"},
        },
    },
}

if _HAS_XGBOOST:
    REGRESSION_ALGORITHMS["XGBoost Regressor"] = {
        "class": XGBRegressor,
        "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
            "max_depth": {"type": "int", "min": 1, "max": 20, "default": 6},
            "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
        },
    }


CLUSTERING_ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "KMeans": {
        "class": KMeans,
        "defaults": {"n_clusters": 3, "random_state": 42, "n_init": 10},
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 30, "default": 3},
        },
    },
    "DBSCAN": {
        "class": DBSCAN,
        "defaults": {"eps": 0.5, "min_samples": 5},
        "params": {
            "eps": {"type": "float", "min": 0.01, "max": 10, "default": 0.5},
            "min_samples": {"type": "int", "min": 1, "max": 50, "default": 5},
        },
    },
}

ALL_ALGORITHMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "classification": CLASSIFICATION_ALGORITHMS,
    "regression": REGRESSION_ALGORITHMS,
    "clustering": CLUSTERING_ALGORITHMS,
}


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

class TrainResult:
    """Container for model training results."""

    def __init__(
        self,
        task: str,
        algorithm: str,
        model: Any,
        metrics: Dict[str, Any],
        feature_names: List[str],
        target_name: str,
        feature_importances: Optional[np.ndarray] = None,
        confusion_mat: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> None:
        self.task = task
        self.algorithm = algorithm
        self.model = model
        self.metrics = metrics
        self.feature_names = feature_names
        self.target_name = target_name
        self.feature_importances = feature_importances
        self.confusion_mat = confusion_mat
        self.y_test = y_test
        self.y_pred = y_pred

    def to_dict(self) -> Dict[str, Any]:
        """Serialise (without the model object)."""
        return {
            "task": self.task,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
        }


# ---------------------------------------------------------------------------
# MLLab
# ---------------------------------------------------------------------------

class MLLab:
    """High-level API for training, evaluating, and exporting ML models."""

    def __init__(self) -> None:
        self.results: List[TrainResult] = []
        self._label_encoders: Dict[str, LabelEncoder] = {}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _parse_param(value: str, ptype: str) -> Any:
        """Convert a string parameter value to the correct Python type."""
        if ptype == "int":
            return int(float(value))
        elif ptype == "float":
            return float(value)
        elif ptype == "int_or_none":
            if value.strip().lower() in ("none", ""):
                return None
            return int(float(value))
        elif ptype == "choice":
            return value
        return value

    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Encode non-numeric features and return a numeric DataFrame."""
        out = df[feature_cols].copy()
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                le = LabelEncoder()
                out[col] = le.fit_transform(out[col].astype(str))
                self._label_encoders[col] = le
        out = out.fillna(0)
        return out

    # -- train ---------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task: str,
        algorithm_name: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
    ) -> TrainResult:
        """Train a model and compute evaluation metrics.

        Args:
            df: Full DataFrame.
            target_col: Name of the target column.
            feature_cols: List of feature column names.
            task: 'classification', 'regression', or 'clustering'.
            algorithm_name: Key in the corresponding algorithm registry.
            params: Hyperparameter overrides.
            test_size: Fraction reserved for testing (not used for clustering).

        Returns:
            A TrainResult with model, metrics, and optional feature importances.
        """
        algo_registry = ALL_ALGORITHMS[task]
        algo_info = algo_registry[algorithm_name]
        AlgoClass = algo_info["class"]

        # Merge defaults with user overrides
        merged = {**algo_info["defaults"]}
        param_specs = algo_info["params"]
        for k, v in params.items():
            if k in param_specs:
                merged[k] = self._parse_param(str(v), param_specs[k]["type"])
            else:
                merged[k] = v

        X = self._prepare_features(df, feature_cols)

        if task in ("classification", "regression"):
            y = df[target_col].copy()
            if task == "classification" and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
                self._label_encoders[target_col] = le

            y = y.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = AlgoClass(**merged)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = self._compute_metrics(task, y_test, y_pred, model, X_test)
            fi = self._get_feature_importances(model, feature_cols)
            cm = confusion_matrix(y_test, y_pred) if task == "classification" else None

            result = TrainResult(
                task=task,
                algorithm=algorithm_name,
                model=model,
                metrics=metrics,
                feature_names=feature_cols,
                target_name=target_col,
                feature_importances=fi,
                confusion_mat=cm,
                y_test=np.array(y_test),
                y_pred=np.array(y_pred),
            )

        else:  # clustering
            model = AlgoClass(**merged)
            labels = model.fit_predict(X)
            metrics = self._compute_clustering_metrics(X, labels, model)
            result = TrainResult(
                task=task,
                algorithm=algorithm_name,
                model=model,
                metrics=metrics,
                feature_names=feature_cols,
                target_name="cluster",
                y_pred=labels,
            )

        self.results.append(result)
        return result

    # -- metrics -------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        task: str,
        y_test: Any,
        y_pred: Any,
        model: Any,
        X_test: Any,
    ) -> Dict[str, float]:
        """Compute metrics for classification or regression."""
        if task == "classification":
            n_classes = len(np.unique(y_test))
            avg = "binary" if n_classes == 2 else "weighted"
            m: Dict[str, float] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            }
            # ROC-AUC (only for binary + probability)
            if n_classes == 2 and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)[:, 1]
                    m["roc_auc"] = float(roc_auc_score(y_test, proba))
                except Exception:
                    pass
            return m
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            return {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, y_pred)),
            }

    @staticmethod
    def _compute_clustering_metrics(
        X: pd.DataFrame,
        labels: np.ndarray,
        model: Any,
    ) -> Dict[str, float]:
        """Compute clustering metrics."""
        m: Dict[str, float] = {}
        n_labels = len(set(labels) - {-1})
        if n_labels > 1 and n_labels < len(X):
            try:
                m["silhouette"] = float(silhouette_score(X, labels))
            except Exception:
                pass
        if hasattr(model, "inertia_"):
            m["inertia"] = float(model.inertia_)
        m["n_clusters"] = float(n_labels)
        return m

    @staticmethod
    def _get_feature_importances(
        model: Any,
        feature_names: List[str],
    ) -> Optional[np.ndarray]:
        """Extract feature importances, if available."""
        if hasattr(model, "feature_importances_"):
            return np.array(model.feature_importances_)
        if hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            return np.abs(coef)
        return None

    # -- export --------------------------------------------------------------

    @staticmethod
    def export_model(model: Any, path: Path) -> None:
        """Pickle-dump a trained model to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(model, fh)

    @staticmethod
    def load_model(path: Path) -> Any:
        """Load a pickled model."""
        with open(path, "rb") as fh:
            return pickle.load(fh)
