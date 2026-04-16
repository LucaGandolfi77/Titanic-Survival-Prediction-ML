"""
Supervised baseline — lower bound reference.

Trains a RandomForestClassifier on the labeled set only,
ignoring all unlabeled data. Every SSL strategy that scores
below this baseline is actively harmful.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import CFG


def supervised_baseline(
    X_labeled: NDArray[np.floating],
    y_labeled: NDArray[np.integer],
    X_test: NDArray[np.floating],
    y_test: NDArray[np.integer],
    random_state: int = 42,
) -> Dict[str, float]:
    """Train RF on labeled data only and evaluate on test set."""
    clf = RandomForestClassifier(
        n_estimators=CFG.RF_N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_labeled, y_labeled)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }


if __name__ == "__main__":
    from data.loaders import load_dataset
    from data.labeled_splitter import labeled_unlabeled_split
    X, y, _ = load_dataset("iris")
    rng = np.random.default_rng(42)
    Xl, yl, Xu, yu = labeled_unlabeled_split(X, y, 0.10, rng)
    result = supervised_baseline(Xl, yl, Xu, yu)
    print(f"Supervised baseline: {result}")
