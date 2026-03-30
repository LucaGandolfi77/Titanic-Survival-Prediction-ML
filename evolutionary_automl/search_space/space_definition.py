"""
Full search space definition for the evolutionary pipeline optimizer.

The search space encodes preprocessing, feature selection, dimensionality
reduction, classifier selection, and per-classifier hyperparameters as a
fixed-length chromosome. Genes use normalized [0,1] floats that are decoded
to actual parameter values during pipeline construction.

Gene layout (13 genes total):
    Gene  0: scaler_type           (categorical, 4 options)
    Gene  1: feature_sel_type      (categorical, 4 options)
    Gene  2: k_features_ratio      (float [0,1] → mapped to k in [1, n_features])
    Gene  3: dim_reduction_type    (categorical, 3 options)
    Gene  4: classifier_type       (categorical, 7 options)
    Genes 5-12: classifier hyperparameters (float [0,1], decoded per classifier)
"""
from __future__ import annotations

SCALER_OPTIONS = ["none", "standard", "minmax", "robust"]

FEATURE_SEL_OPTIONS = ["none", "select_k_best", "select_from_model", "variance_threshold"]

DIM_REDUCTION_OPTIONS = ["none", "pca", "truncated_svd"]

CLASSIFIER_OPTIONS = [
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svc",
    "knn",
    "logistic_regression",
    "mlp",
]

# Per-classifier hyperparameter mappings.
# Each entry maps gene indices [5..12] to hyperparameter name + decode range.
# Unused genes for a given classifier are simply ignored.

HYPERPARAMETER_SPACE = {
    "decision_tree": {
        5: ("max_depth", "int", 1, 30),
        6: ("min_samples_split", "int", 2, 40),
        7: ("min_samples_leaf", "int", 1, 20),
        8: ("criterion", "cat", ["gini", "entropy"]),
    },
    "random_forest": {
        5: ("n_estimators", "int", 10, 300),
        6: ("max_depth", "int", 2, 30),
        7: ("min_samples_split", "int", 2, 20),
        8: ("min_samples_leaf", "int", 1, 10),
        9: ("max_features", "cat", ["sqrt", "log2", None]),
    },
    "gradient_boosting": {
        5: ("n_estimators", "int", 50, 300),
        6: ("learning_rate", "float", 0.01, 0.5),
        7: ("max_depth", "int", 2, 10),
        8: ("subsample", "float", 0.5, 1.0),
        9: ("min_samples_split", "int", 2, 20),
    },
    "svc": {
        5: ("C", "log_float", 0.01, 100.0),
        6: ("kernel", "cat", ["rbf", "linear", "poly"]),
        7: ("gamma", "cat", ["scale", "auto"]),
        8: ("degree", "int", 2, 5),
    },
    "knn": {
        5: ("n_neighbors", "int", 1, 30),
        6: ("weights", "cat", ["uniform", "distance"]),
        7: ("metric", "cat", ["euclidean", "manhattan", "minkowski"]),
        8: ("p", "int", 1, 5),
    },
    "logistic_regression": {
        5: ("C", "log_float", 0.01, 100.0),
        6: ("solver", "cat", ["lbfgs", "liblinear", "saga"]),
        7: ("max_iter", "int", 100, 1000),
        8: ("penalty", "cat", ["l2", "l1", "none"]),
    },
    "mlp": {
        5: ("hidden_layer_sizes_0", "int", 16, 256),
        6: ("hidden_layer_sizes_1", "int", 0, 128),
        7: ("learning_rate_init", "log_float", 0.0001, 0.1),
        8: ("alpha", "log_float", 0.0001, 0.1),
        9: ("activation", "cat", ["relu", "tanh"]),
        10: ("max_iter", "int", 100, 500),
    },
}


def decode_gene(value: float, gene_spec: tuple) -> object:
    """Decode a normalized [0,1] gene value into its actual parameter value.

    Args:
        value: float in [0, 1]
        gene_spec: (name, type, *args) specification tuple
    """
    name, gtype = gene_spec[0], gene_spec[1]
    v = max(0.0, min(1.0, value))

    if gtype == "int":
        lo, hi = gene_spec[2], gene_spec[3]
        return int(round(lo + v * (hi - lo)))
    elif gtype == "float":
        lo, hi = gene_spec[2], gene_spec[3]
        return lo + v * (hi - lo)
    elif gtype == "log_float":
        import math
        lo, hi = gene_spec[2], gene_spec[3]
        log_lo, log_hi = math.log10(lo), math.log10(hi)
        return 10 ** (log_lo + v * (log_hi - log_lo))
    elif gtype == "cat":
        options = gene_spec[2]
        idx = int(v * (len(options) - 1) + 0.5)
        idx = max(0, min(len(options) - 1, idx))
        return options[idx]
    else:
        raise ValueError(f"Unknown gene type: {gtype}")


def categorical_index(value: float, n_options: int) -> int:
    """Map a [0,1] float to an integer index in [0, n_options-1]."""
    idx = int(value * (n_options - 1) + 0.5)
    return max(0, min(n_options - 1, idx))


if __name__ == "__main__":
    print("Scaler options:", SCALER_OPTIONS)
    print("Classifiers:", CLASSIFIER_OPTIONS)
    for clf, params in HYPERPARAMETER_SPACE.items():
        print(f"\n{clf}:")
        for gene_idx, spec in params.items():
            print(f"  Gene {gene_idx}: {spec}")
