"""
Heterogeneous Ensemble — Soft Voting
======================================
VotingClassifier with probability-based combination (soft voting).
All base learners must support predict_proba.
"""

from __future__ import annotations

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_soft_voting(random_state: int = 42) -> VotingClassifier:
    estimators = [
        ("dt", DecisionTreeClassifier(max_depth=None, random_state=random_state)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=random_state)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("nb", GaussianNB()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name
    from sklearn.model_selection import cross_val_score

    X, y, _ = get_dataset_by_name("iris")
    clf = build_soft_voting()
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"SoftVoting  acc={scores.mean():.3f} ± {scores.std():.3f}")
