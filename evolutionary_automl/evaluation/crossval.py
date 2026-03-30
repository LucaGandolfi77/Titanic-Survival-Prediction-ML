"""
Cross-validation utilities with multiple seeds.

Provides stratified k-fold evaluation repeated across multiple random
seeds for robust performance estimation of a pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ..config import CFG


def multi_seed_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    seeds: Tuple[int, ...] = CFG.RANDOM_SEEDS,
    cv_folds: int = CFG.CV_FOLDS,
    scoring: str = "f1_macro",
) -> Dict[str, object]:
    """Run stratified k-fold CV with multiple random seeds.

    Args:
        pipeline: sklearn Pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.
        seeds: Tuple of random seeds.
        cv_folds: Number of folds per seed.
        scoring: Scoring metric name.

    Returns:
        Dict with per_seed_scores (list of arrays), mean_scores (per seed),
        overall_mean, overall_std.
    """
    per_seed_scores = []
    mean_scores = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y, cv=skf, scoring=scoring, n_jobs=1, error_score=0.0
        )
        per_seed_scores.append(scores.tolist())
        mean_scores.append(float(np.mean(scores)))

    return {
        "per_seed_scores": per_seed_scores,
        "mean_scores": mean_scores,
        "overall_mean": float(np.mean(mean_scores)),
        "overall_std": float(np.std(mean_scores, ddof=1)) if len(mean_scores) > 1 else 0.0,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=0)),
    ])
    result = multi_seed_cv(pipe, X, y, seeds=(42, 7, 13))
    print(f"Mean scores per seed: {result['mean_scores']}")
    print(f"Overall: {result['overall_mean']:.4f} ± {result['overall_std']:.4f}")
