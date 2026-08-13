"""exercise2b_overfit.py — Overfitting control via min_samples_leaf (M).

Sweep M from 1 to 30, record train/test accuracy, depth, #leaves.
Identify optimal M and visualise the bias-variance tradeoff.

Outputs:
  - outputs/ex2b_M_sweep_results.csv
  - outputs/ex2b_overfitting_curve.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data_loader import load_arff, arff_features_and_label


def run(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n" + "=" * 64)
    print("Exercise 2b — Overfitting Control (min_samples_leaf sweep)")
    print("=" * 64)

    df_train = load_arff(f"{data_dir}/Bigtest1_104.arff")
    df_test = load_arff(f"{data_dir}/Bigtest2_104.arff")
    X_train, y_train = arff_features_and_label(df_train)
    X_test, y_test = arff_features_and_label(df_test)

    m_values = list(range(1, 31))
    records: list[dict] = []

    print(f"\n  {'M':>4} {'Train':>8} {'Test':>8} {'Depth':>6} {'Leaves':>7}")
    print("  " + "-" * 38)

    for m in m_values:
        dt = DecisionTreeClassifier(min_samples_leaf=m, random_state=42)
        dt.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        depth = dt.tree_.max_depth
        leaves = dt.tree_.n_leaves

        records.append({
            "M": m,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "depth": depth,
            "leaves": leaves,
        })
        print(f"  {m:>4} {train_acc:>8.2%} {test_acc:>8.2%} "
              f"{depth:>6} {leaves:>7}")

    df_results = pd.DataFrame(records)
    df_results.to_csv(f"{out_dir}/ex2b_M_sweep_results.csv", index=False)
    print(f"\n  [csv] Saved: {out_dir}/ex2b_M_sweep_results.csv")

    # Optimal M
    best_idx = df_results["test_accuracy"].idxmax()
    best_row = df_results.iloc[best_idx]
    best_m = int(best_row["M"])
    print(f"\n  ★ Optimal M = {best_m}  "
          f"(test acc = {best_row['test_accuracy']:.2%}, "
          f"depth = {int(best_row['depth'])}, "
          f"leaves = {int(best_row['leaves'])})")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Subplot 1: accuracy curves
    ax1.plot(m_values, df_results["train_accuracy"], "o-", label="Train", ms=4)
    ax1.plot(m_values, df_results["test_accuracy"], "s-", label="Test", ms=4)
    ax1.axvline(best_m, color="red", linestyle="--", alpha=0.7,
                label=f"Optimal M={best_m}")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Train / Test Accuracy vs min_samples_leaf (M)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: tree complexity
    ax2.plot(m_values, df_results["depth"], "^-", color="purple",
             label="Depth", ms=4)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(m_values, df_results["leaves"], "D-", color="orange",
                  label="Leaves", ms=4)
    ax2.axvline(best_m, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("min_samples_leaf (M)")
    ax2.set_ylabel("Tree Depth", color="purple")
    ax2_twin.set_ylabel("Number of Leaves", color="orange")
    ax2.set_title("Tree Complexity vs M")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/ex2b_overfitting_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {out_dir}/ex2b_overfitting_curve.png")

    # Discussion
    print(
        "\n  Bias-Variance Tradeoff:"
        "\n  • M=1 (no constraint): the tree grows as deep as possible,"
        "\n    perfectly fitting the training data (high variance, low bias)."
        "\n    Test accuracy suffers because the tree memorises noise."
        "\n  • As M increases, leaves must contain more samples. The tree"
        "\n    becomes shallower and more general (lower variance, higher bias)."
        "\n  • The optimal M balances these two forces: the tree is complex"
        "\n    enough to capture the real patterns but not so deep that it"
        "\n    overfits the training noise."
        f"\n  • Here, the sweet spot is M = {best_m}."
    )
