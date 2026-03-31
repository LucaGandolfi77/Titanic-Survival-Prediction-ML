"""
Ensemble Factory
=================
Build any method from the study by name.  Returns a fitted-ready
classifier and its canonical label.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import CFG
from ensembles.heterogeneous import build_hard_voting
from ensembles.homogeneous import (
    build_adaboost,
    build_bagging,
    build_gradient_boosting,
    build_random_forest,
)
from ensembles.soft_voting import build_soft_voting


def build_method(
    name: str,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    random_state: int = 42,
    **kwargs: Any,
) -> ClassifierMixin:
    """Instantiate a classifier by canonical name.

    Valid names: bagging, random_forest, adaboost, gradient_boosting,
                hard_voting, soft_voting, decision_tree, logistic_regression.
    """
    if name == "bagging":
        return build_bagging(n_estimators=n_estimators, random_state=random_state)
    elif name == "random_forest":
        return build_random_forest(n_estimators=n_estimators, random_state=random_state, **kwargs)
    elif name == "adaboost":
        return build_adaboost(n_estimators=n_estimators, random_state=random_state, **kwargs)
    elif name == "gradient_boosting":
        return build_gradient_boosting(n_estimators=n_estimators, random_state=random_state, **kwargs)
    elif name == "hard_voting":
        return build_hard_voting(random_state=random_state)
    elif name == "soft_voting":
        return build_soft_voting(random_state=random_state)
    elif name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    elif name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {name}")


METHOD_LABELS: Dict[str, str] = {
    "bagging": "Bagging",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Grad. Boosting",
    "hard_voting": "Hard Voting",
    "soft_voting": "Soft Voting",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Reg.",
}


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


if __name__ == "__main__":
    for name in CFG.METHOD_NAMES:
        clf = build_method(name, n_estimators=10)
        print(f"{method_label(name):20s}  {type(clf).__name__}")
