"""
counterfactuals.py – "What-if" counterfactual analysis.

Given a data point and its prediction, find the minimal feature changes
that would flip the model's decision. Uses a simple greedy search approach.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def find_counterfactuals(
    estimator: BaseEstimator,
    instance: pd.DataFrame,
    desired_class: int | str,
    X_train: pd.DataFrame,
    n_counterfactuals: int = 3,
    features_to_vary: list[str] | None = None,
    max_iterations: int = 1000,
) -> pd.DataFrame:
    """Find counterfactual examples using a greedy perturbation search.

    Parameters
    ----------
    estimator : fitted classifier
    instance : single-row DataFrame
    desired_class : the target class we want the model to predict
    X_train : training data (used to sample realistic feature values)
    n_counterfactuals : how many to return
    features_to_vary : if None, all features can change

    Returns
    -------
    DataFrame with counterfactual rows + metadata columns:
        - _changed_features: list of features changed
        - _n_changes: number of features changed
        - _distance: L1 distance from original (normalised)
    """
    row = instance.copy()
    cols = features_to_vary or X_train.columns.tolist()

    # Normalisation ranges for distance calculation
    ranges = {}
    for c in cols:
        r = X_train[c].max() - X_train[c].min()
        ranges[c] = r if r > 0 else 1.0

    found: List[Dict] = []
    rng = np.random.RandomState(42)

    for _ in range(max_iterations):
        if len(found) >= n_counterfactuals:
            break

        candidate = row.copy()

        # Randomly decide how many features to change (1–len(cols)//2)
        n_change = rng.randint(1, max(2, len(cols) // 2 + 1))
        change_cols = rng.choice(cols, size=n_change, replace=False)

        changed = []
        for c in change_cols:
            # Pick a random value from the training distribution
            candidate[c] = rng.choice(X_train[c].values)
            changed.append(c)

        pred = estimator.predict(candidate)[0]
        if pred == desired_class:
            dist = sum(abs(candidate[c].values[0] - row[c].values[0]) / ranges[c] for c in changed)
            entry = candidate.iloc[0].to_dict()
            entry["_changed_features"] = ", ".join(changed)
            entry["_n_changes"] = len(changed)
            entry["_distance"] = round(float(dist), 4)
            found.append(entry)

    if not found:
        return pd.DataFrame()

    cf_df = pd.DataFrame(found)
    cf_df = cf_df.sort_values("_distance").head(n_counterfactuals).reset_index(drop=True)
    return cf_df


def what_if_analysis(
    estimator: BaseEstimator,
    instance: pd.DataFrame,
    feature: str,
    values: np.ndarray | list,
) -> pd.DataFrame:
    """Sweep a single feature across *values* and record predictions.

    Returns a tidy DataFrame: feature_value, prediction, [prob_0, prob_1, …]
    """
    records = []
    for v in values:
        modified = instance.copy()
        modified[feature] = v

        pred = estimator.predict(modified)[0]
        record = {"feature_value": v, "prediction": pred}

        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(modified)[0]
            for i, p in enumerate(proba):
                record[f"prob_class_{i}"] = round(float(p), 4)

        records.append(record)

    return pd.DataFrame(records)
