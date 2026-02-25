"""
evaluator.py – Model training and evaluation utilities.
Provides a unified interface for classification, regression,
clustering, feature selection, and association rules.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    StratifiedKFold,
    KFold,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Classifiers ───────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Regressors ────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB_REG = True
except ImportError:
    HAS_XGB_REG = False

# ── Clustering ────────────────────────────────────────────────────────
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    MiniBatchKMeans,
)

# ── Feature selection ─────────────────────────────────────────────────
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)

# ── Association rules ─────────────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

SEED = 42


# ======================================================================
# Classifier catalogue
# ======================================================================
def get_classifiers() -> dict[str, Any]:
    """Return a dict of name -> (constructor, default_params)."""
    clfs: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEED),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=SEED),
    }
    if HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=100, eval_metric="logloss", random_state=SEED, n_jobs=-1
        )
    return clfs


def get_regressors() -> dict[str, Any]:
    regs: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=2000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=SEED),
        "Bagging": BaggingRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
    }
    if HAS_XGB_REG:
        regs["XGBoost"] = XGBRegressor(
            n_estimators=100, random_state=SEED, n_jobs=-1
        )
    return regs


def get_clusterers() -> dict[str, Any]:
    return {
        "K-Means": KMeans(n_clusters=3, random_state=SEED, n_init=10),
        "Mini-Batch K-Means": MiniBatchKMeans(n_clusters=3, random_state=SEED, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
        "Mean Shift": MeanShift(),
    }


# ======================================================================
# Evaluation helpers
# ======================================================================
class ClassificationResult:
    def __init__(self, name, model, y_true, y_pred, train_time, classes, cv_scores=None):
        self.name = name
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.train_time = train_time
        self.classes = classes
        self.cv_scores = cv_scores
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        self.f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        self.cm = confusion_matrix(y_true, y_pred)
        self.report = classification_report(y_true, y_pred, zero_division=0)

    def summary_line(self) -> str:
        s = (f"{self.name:<28s}  Acc={self.accuracy:.4f}  "
             f"Prec={self.precision:.4f}  Rec={self.recall:.4f}  "
             f"F1={self.f1:.4f}  Time={self.train_time:.2f}s")
        if self.cv_scores is not None:
            s += f"  CV={self.cv_scores.mean():.4f}+/-{self.cv_scores.std():.4f}"
        return s


class RegressionResult:
    def __init__(self, name, model, y_true, y_pred, train_time, cv_scores=None):
        self.name = name
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.train_time = train_time
        self.cv_scores = cv_scores
        self.r2 = r2_score(y_true, y_pred)
        self.rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.mae = mean_absolute_error(y_true, y_pred)

    def summary_line(self) -> str:
        s = (f"{self.name:<28s}  R2={self.r2:.4f}  "
             f"RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  "
             f"Time={self.train_time:.2f}s")
        if self.cv_scores is not None:
            s += f"  CV-R2={self.cv_scores.mean():.4f}+/-{self.cv_scores.std():.4f}"
        return s


class ClusterResult:
    def __init__(self, name, model, labels, X, train_time):
        self.name = name
        self.model = model
        self.labels = labels
        self.X = X
        self.train_time = train_time
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)
        if n_labels >= 2 and n_labels < len(labels):
            sample = min(5000, len(X))
            idx = np.random.RandomState(SEED).choice(len(X), sample, replace=False)
            self.silhouette = silhouette_score(X[idx], labels[idx])
        else:
            self.silhouette = float("nan")
        self.n_clusters = n_labels
        unique, counts = np.unique(labels, return_counts=True)
        self.cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    def summary_line(self) -> str:
        return (f"{self.name:<28s}  Clusters={self.n_clusters}  "
                f"Silhouette={self.silhouette:.4f}  Time={self.train_time:.2f}s")


# ======================================================================
# Runner functions
# ======================================================================
def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model,
    eval_mode: str = "split",
    test_size: float = 0.2,
    cv_folds: int = 10,
    scale: bool = True,
    classes: list | None = None,
) -> ClassificationResult:
    """Run a classification experiment."""
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if eval_mode == "cv":
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        t0 = time.time()
        y_pred = cross_val_predict(model, X, y, cv=skf)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")
        model.fit(X, y)
        elapsed = time.time() - t0
        return ClassificationResult(model_name, model, y, y_pred, elapsed, classes, cv_scores)
    elif eval_mode == "training_set":
        t0 = time.time()
        model.fit(X, y)
        y_pred = model.predict(X)
        elapsed = time.time() - t0
        return ClassificationResult(model_name, model, y, y_pred, elapsed, classes)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y
        )
        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        elapsed = time.time() - t0
        return ClassificationResult(model_name, model, y_te, y_pred, elapsed, classes)


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model,
    eval_mode: str = "split",
    test_size: float = 0.2,
    cv_folds: int = 10,
    scale: bool = True,
) -> RegressionResult:
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if eval_mode == "cv":
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        t0 = time.time()
        y_pred = cross_val_predict(model, X, y, cv=kf)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        model.fit(X, y)
        elapsed = time.time() - t0
        return RegressionResult(model_name, model, y, y_pred, elapsed, cv_scores)
    elif eval_mode == "training_set":
        t0 = time.time()
        model.fit(X, y)
        y_pred = model.predict(X)
        elapsed = time.time() - t0
        return RegressionResult(model_name, model, y, y_pred, elapsed)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=SEED
        )
        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        elapsed = time.time() - t0
        return RegressionResult(model_name, model, y_te, y_pred, elapsed)


def run_clustering(
    X: np.ndarray,
    model_name: str,
    model,
    scale: bool = True,
) -> ClusterResult:
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return ClusterResult(model_name, model, labels, X, elapsed)


# ======================================================================
# Feature selection
# ======================================================================
def select_features_kbest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 10,
    task: str = "classification",
) -> list[tuple[str, float]]:
    func = f_classif if task == "classification" else f_regression
    sel = SelectKBest(func, k=min(k, X.shape[1]))
    sel.fit(X, y)
    scores = sel.scores_
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return ranked


def select_features_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str = "classification",
) -> list[tuple[str, float]]:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X, y)
    imp = model.feature_importances_
    ranked = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    return ranked


def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features: int = 10,
    task: str = "classification",
) -> list[tuple[str, int]]:
    if task == "classification":
        estimator = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]))
    rfe.fit(X, y)
    ranked = sorted(zip(feature_names, rfe.ranking_), key=lambda x: x[1])
    return ranked
