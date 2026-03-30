"""
Pipeline builder: converts a decoded chromosome into a scikit-learn Pipeline.

Handles incompatible combinations gracefully by falling back to safe defaults
(e.g., skipping sparse-incompatible steps, adjusting PCA components).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .space_definition import (
    CLASSIFIER_OPTIONS,
    DIM_REDUCTION_OPTIONS,
    FEATURE_SEL_OPTIONS,
    HYPERPARAMETER_SPACE,
    SCALER_OPTIONS,
    categorical_index,
    decode_gene,
)


def _build_scaler(gene0: float) -> Optional[Any]:
    idx = categorical_index(gene0, len(SCALER_OPTIONS))
    choice = SCALER_OPTIONS[idx]
    if choice == "none":
        return None
    elif choice == "standard":
        return StandardScaler()
    elif choice == "minmax":
        return MinMaxScaler()
    elif choice == "robust":
        return RobustScaler()


def _build_feature_selector(gene1: float, gene2: float, n_features: int) -> Optional[Any]:
    idx = categorical_index(gene1, len(FEATURE_SEL_OPTIONS))
    choice = FEATURE_SEL_OPTIONS[idx]

    if choice == "none":
        return None

    k = max(1, int(round(gene2 * (n_features - 1) + 1)))
    k = min(k, n_features)

    if choice == "select_k_best":
        return SelectKBest(score_func=f_classif, k=k)
    elif choice == "select_from_model":
        return SelectFromModel(
            ExtraTreesClassifier(n_estimators=50, random_state=0),
            max_features=k,
        )
    elif choice == "variance_threshold":
        return VarianceThreshold(threshold=gene2 * 0.1)
    return None


def _build_dim_reduction(gene3: float, n_features: int) -> Optional[Any]:
    idx = categorical_index(gene3, len(DIM_REDUCTION_OPTIONS))
    choice = DIM_REDUCTION_OPTIONS[idx]

    if choice == "none":
        return None

    n_components = max(2, int(round(0.3 * n_features + gene3 * 0.5 * n_features)))
    n_components = min(n_components, n_features - 1) if n_features > 2 else n_features

    if choice == "pca":
        return PCA(n_components=n_components)
    elif choice == "truncated_svd":
        return TruncatedSVD(n_components=max(1, n_components))
    return None


def _decode_classifier_params(clf_name: str, genes: List[float]) -> Dict[str, Any]:
    """Decode genes 5-12 into classifier-specific hyperparameters."""
    params = {}
    hp_space = HYPERPARAMETER_SPACE.get(clf_name, {})
    for gene_idx, spec in hp_space.items():
        gene_val = genes[gene_idx] if gene_idx < len(genes) else 0.5
        params[spec[0]] = decode_gene(gene_val, spec)
    return params


def _build_classifier(gene4: float, genes: List[float], random_state: int = 0) -> Any:
    idx = categorical_index(gene4, len(CLASSIFIER_OPTIONS))
    clf_name = CLASSIFIER_OPTIONS[idx]
    params = _decode_classifier_params(clf_name, genes)

    if clf_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            criterion=params.get("criterion", "gini"),
            random_state=random_state,
        )
    elif clf_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=1,
        )
    elif clf_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 1.0),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=random_state,
        )
    elif clf_name == "svc":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            degree=params.get("degree", 3),
            random_state=random_state,
        )
    elif clf_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski"),
            p=params.get("p", 2),
            n_jobs=1,
        )
    elif clf_name == "logistic_regression":
        penalty = params.get("penalty", "l2")
        solver = params.get("solver", "lbfgs")
        # Fix incompatible solver/penalty combinations
        if penalty == "l1":
            solver = "saga"
        if penalty == "none":
            penalty = None
            solver = "lbfgs"
        if solver == "liblinear":
            solver = "lbfgs"
        return LogisticRegression(
            C=params.get("C", 1.0),
            solver=solver,
            max_iter=params.get("max_iter", 200),
            penalty=penalty,
            random_state=random_state,
        )
    elif clf_name == "mlp":
        h0 = params.get("hidden_layer_sizes_0", 64)
        h1 = params.get("hidden_layer_sizes_1", 0)
        hidden = (h0,) if h1 == 0 else (h0, h1)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            learning_rate_init=params.get("learning_rate_init", 0.001),
            alpha=params.get("alpha", 0.0001),
            activation=params.get("activation", "relu"),
            max_iter=params.get("max_iter", 200),
            random_state=random_state,
        )
    raise ValueError(f"Unknown classifier: {clf_name}")


def build_pipeline(
    chromosome: List[float],
    n_features: int,
    random_state: int = 0,
) -> Pipeline:
    """Construct a sklearn Pipeline from a chromosome.

    Args:
        chromosome: List of 13 floats, each in [0, 1].
        n_features: Number of features in the dataset.
        random_state: Random state for reproducibility.

    Returns:
        A configured sklearn Pipeline ready for fit/predict.
    """
    steps = []

    scaler = _build_scaler(chromosome[0])
    if scaler is not None:
        steps.append(("scaler", scaler))

    feat_sel = _build_feature_selector(chromosome[1], chromosome[2], n_features)
    if feat_sel is not None:
        steps.append(("feature_selection", feat_sel))

    dim_red = _build_dim_reduction(chromosome[3], n_features)
    if dim_red is not None:
        steps.append(("dim_reduction", dim_red))

    clf = _build_classifier(chromosome[4], chromosome, random_state)
    steps.append(("classifier", clf))

    return Pipeline(steps)


def describe_pipeline(chromosome: List[float], n_features: int) -> str:
    """Return a human-readable description of the pipeline encoded by a chromosome."""
    parts = []

    scaler_idx = categorical_index(chromosome[0], len(SCALER_OPTIONS))
    parts.append(f"Scaler={SCALER_OPTIONS[scaler_idx]}")

    fs_idx = categorical_index(chromosome[1], len(FEATURE_SEL_OPTIONS))
    parts.append(f"FeatureSel={FEATURE_SEL_OPTIONS[fs_idx]}")

    dr_idx = categorical_index(chromosome[3], len(DIM_REDUCTION_OPTIONS))
    parts.append(f"DimRed={DIM_REDUCTION_OPTIONS[dr_idx]}")

    clf_idx = categorical_index(chromosome[4], len(CLASSIFIER_OPTIONS))
    clf_name = CLASSIFIER_OPTIONS[clf_idx]
    params = _decode_classifier_params(clf_name, chromosome)
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    parts.append(f"Clf={clf_name}({param_str})")

    return " → ".join(parts)


if __name__ == "__main__":
    test_chromosome = [0.3, 0.5, 0.5, 0.0, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pipe = build_pipeline(test_chromosome, n_features=30)
    print("Pipeline steps:", [s[0] for s in pipe.steps])
    print("Description:", describe_pipeline(test_chromosome, 30))
