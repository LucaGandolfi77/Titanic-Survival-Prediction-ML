"""exercise1_knn.py — kNN on the circle/square classification problem.

Outputs:
  - Accuracy for k = 1, 3, 5, 7, 9, 11 on the test set
  - Best-k evaluation on circleall.arff
  - Classification reports and confusion matrices
  - outputs/ex1_accuracy_vs_k.png
  - outputs/ex1_decision_boundaries.png
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_loader import load_arff, arff_features_and_label
from utils import print_metrics, plot_decision_boundary


# ------------------------------------------------------------------
# Scratch kNN implementation
# ------------------------------------------------------------------

class KNNScratch:
    """k-Nearest Neighbors from scratch (Euclidean distance, majority vote)."""

    def __init__(self, k: int = 3):
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        self._X_train = X.copy()
        self._y_train = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self._X_train - x) ** 2, axis=1))
            idx = np.argsort(dists)[: self.k]
            neighbours = self._y_train[idx]
            # Majority vote
            labels, counts = np.unique(neighbours, return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)


# ------------------------------------------------------------------
# Main exercise logic
# ------------------------------------------------------------------

def run(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("=" * 64)
    print("Exercise 1 — kNN on Circle/Square Problem")
    print("=" * 64)

    # Load data
    df_train = load_arff(f"{data_dir}/circletrain.arff")
    df_test = load_arff(f"{data_dir}/circletest.arff")
    df_all = load_arff(f"{data_dir}/circleall.arff")

    X_train, y_train = arff_features_and_label(df_train)
    X_test, y_test = arff_features_and_label(df_test)
    X_all, y_all = arff_features_and_label(df_all)

    k_values = [1, 3, 5, 7, 9, 11]
    accs_scratch: dict[int, float] = {}
    accs_sklearn: dict[int, float] = {}

    print("\n  k   Scratch Acc   Sklearn Acc")
    print("  " + "-" * 35)

    for k in k_values:
        # Scratch
        knn_s = KNNScratch(k=k).fit(X_train, y_train)
        pred_s = knn_s.predict(X_test)
        acc_s = accuracy_score(y_test, pred_s)
        accs_scratch[k] = acc_s

        # Sklearn
        knn_sk = KNeighborsClassifier(n_neighbors=k)
        knn_sk.fit(X_train, y_train)
        pred_sk = knn_sk.predict(X_test)
        acc_sk = accuracy_score(y_test, pred_sk)
        accs_sklearn[k] = acc_sk

        print(f"  {k:>2}      {acc_s:.2%}        {acc_sk:.2%}")

    # Best k (by sklearn accuracy on test/validation set)
    best_k = max(accs_sklearn, key=accs_sklearn.get)  # type: ignore[arg-type]
    print(f"\n  ★ Best k = {best_k} (test accuracy = {accs_sklearn[best_k]:.2%})")

    # Evaluate best k on circleall
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    pred_all = best_model.predict(X_all)

    print(f"\n  --- Best k={best_k} evaluated on circleall ---")
    print_metrics(y_all, pred_all, f"kNN k={best_k} circleall")

    # Full report per k on test set
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        pred = knn.predict(X_test)
        print(f"\n  --- k={k} on circletest ---")
        print_metrics(y_test, pred, f"kNN k={k}")

    # ---- Plots ----

    # 1) Accuracy vs k
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, [accs_scratch[k] for k in k_values],
            "o--", label="Scratch")
    ax.plot(k_values, [accs_sklearn[k] for k in k_values],
            "s-", label="Sklearn")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("kNN Accuracy vs k (circletest)")
    ax.set_xticks(k_values)
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ex1_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved: {out_dir}/ex1_accuracy_vs_k.png")

    # 2) Decision boundaries for k=1, best_k, k=11
    label_map = {l: i for i, l in enumerate(sorted(set(y_train)))}
    k_show = [1, best_k, 11]
    # Deduplicate if best_k is 1 or 11
    k_show = list(dict.fromkeys(k_show))

    fig, axes = plt.subplots(1, len(k_show), figsize=(5 * len(k_show), 4.5))
    if len(k_show) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_show):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        plot_decision_boundary(clf, X_train, y_train,
                               title=f"kNN (k={k})", ax=ax,
                               label_map=label_map)

    fig.suptitle("Decision Boundaries — kNN", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ex1_decision_boundaries.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {out_dir}/ex1_decision_boundaries.png")
