"""
Fitness metrics: accuracy, F1, training time, number of features, memory.

All metric functions accept a fitted pipeline and/or cross-validation results
and return a scalar value suitable for use as a fitness objective.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def compute_f1_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> float:
    """Compute macro-averaged F1 via stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=1, error_score=0.0
    )
    return float(np.mean(scores))


def measure_training_time(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> float:
    """Measure wall-clock training time in seconds."""
    start = time.perf_counter()
    pipeline.fit(X, y)
    return time.perf_counter() - start


def count_features_used(pipeline: Pipeline, n_original: int) -> int:
    """Count number of features used by the pipeline after transformations.

    Inspects feature selection and dimensionality reduction steps.
    """
    n = n_original
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "get_support"):
            n = int(np.sum(step.get_support()))
        elif hasattr(step, "n_components"):
            if isinstance(step.n_components, int):
                n = min(n, step.n_components)
            elif isinstance(step.n_components, float):
                n = max(1, int(n * step.n_components))
        elif hasattr(step, "n_features_in_") and hasattr(step, "n_features_"):
            pass
    return n


def model_memory_bytes(pipeline: Pipeline) -> int:
    """Estimate memory footprint of the fitted pipeline in bytes."""
    return sys.getsizeof(pipeline)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
    f1 = compute_f1_cv(pipe, X, y)
    print(f"F1 (macro, 5-fold): {f1:.4f}")
    t = measure_training_time(pipe, X, y)
    print(f"Training time: {t:.4f}s")
    mem = model_memory_bytes(pipe)
    print(f"Memory: {mem} bytes")
