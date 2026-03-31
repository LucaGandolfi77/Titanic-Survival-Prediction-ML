"""
Homogeneous Ensembles
======================
Wrappers around sklearn's BaggingClassifier, RandomForestClassifier,
AdaBoostClassifier, and GradientBoostingClassifier with consistent
construction signatures.
"""

from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from config import CFG


def build_bagging(
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    random_state: int = 42,
) -> BaggingClassifier:
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None, random_state=random_state),
        n_estimators=n_estimators,
        bootstrap=True,
        max_samples=1.0,
        max_features=1.0,
        random_state=random_state,
        n_jobs=1,
    )


def build_random_forest(
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    max_features: str | None = "sqrt",
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
    )


def build_adaboost(
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    learning_rate: float = 1.0,
    random_state: int = 42,
) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def build_gradient_boosting(
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name
    from sklearn.model_selection import cross_val_score

    X, y, _ = get_dataset_by_name("iris")
    for name, clf in [
        ("Bagging", build_bagging(n_estimators=10)),
        ("RF", build_random_forest(n_estimators=10)),
        ("AdaBoost", build_adaboost(n_estimators=10)),
        ("GradBoost", build_gradient_boosting(n_estimators=10)),
    ]:
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        print(f"{name:12s}  acc={scores.mean():.3f} ± {scores.std():.3f}")
