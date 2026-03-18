"""exercise2_digits.py — Decision Tree classification on 13×8 digit images.

Tasks:
  1. Random-split experiment (5 seeds, 66/34 split)
  2. Fixed external test set evaluation (Bigtest2)
  3. Tree visualisation

Outputs:
  - outputs/ex2_sample_digits.png
  - outputs/ex2_seed_results.csv
  - outputs/ex2_decision_tree.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_arff, arff_features_and_label
from utils import print_metrics


# ------------------------------------------------------------------
# Visualise sample digits
# ------------------------------------------------------------------

def _plot_sample_digits(X: np.ndarray, y: np.ndarray,
                        out_path: str, n_per_class: int = 2) -> None:
    """Show *n_per_class* sample digit images per class (0-9) in a grid."""
    classes = sorted(set(y))
    n_classes = len(classes)
    fig, axes = plt.subplots(n_per_class, n_classes,
                             figsize=(n_classes * 1.3, n_per_class * 1.6))
    for col, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        chosen = idx[:n_per_class]
        for row, i in enumerate(chosen):
            img = X[i].reshape(13, 8)
            axes[row, col].imshow(img, cmap="gray_r", interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(cls, fontsize=10)
    fig.suptitle("Sample Digits (13×8 binary images)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {out_path}")


# ------------------------------------------------------------------
# Main exercise logic
# ------------------------------------------------------------------

def run(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n" + "=" * 64)
    print("Exercise 2 — Digit Classification (Decision Tree)")
    print("=" * 64)

    df_train_full = load_arff(f"{data_dir}/Bigtest1_104.arff")
    X_train_full, y_train_full = arff_features_and_label(df_train_full)

    # Visualise samples
    _plot_sample_digits(X_train_full, y_train_full,
                        f"{out_dir}/ex2_sample_digits.png")

    # ---- Task 1: Random split experiment ----
    print("\n  --- Task 1: Random Split Experiment (66/34, 5 seeds) ---\n")
    seeds = [0, 1, 2, 3, 4]
    seed_results: list[dict] = []

    print(f"  {'Seed':>6} {'Accuracy':>10}")
    print("  " + "-" * 20)

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_full, y_train_full,
            test_size=0.34, stratify=y_train_full, random_state=seed,
        )
        dt = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, dt.predict(X_te))
        seed_results.append({"seed": seed, "accuracy": acc})
        print(f"  {seed:>6} {acc:>10.2%}")

    accs = [r["accuracy"] for r in seed_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\n  Mean ± Std: {mean_acc:.2%} ± {std_acc:.2%}")

    # Save CSV
    pd.DataFrame(seed_results).to_csv(
        f"{out_dir}/ex2_seed_results.csv", index=False,
    )
    print(f"  [csv] Saved: {out_dir}/ex2_seed_results.csv")

    # ---- Task 2: Fixed external test set ----
    print("\n  --- Task 2: Fixed External Test Set (Bigtest2) ---")

    df_test = load_arff(f"{data_dir}/Bigtest2_104.arff")
    X_test, y_test = arff_features_and_label(df_test)

    dt_full = DecisionTreeClassifier(random_state=42).fit(X_train_full,
                                                          y_train_full)
    pred_test = dt_full.predict(X_test)
    fixed_acc = print_metrics(y_test, pred_test, "Decision Tree (external test)")

    print(f"\n  Comparison:")
    print(f"    Random-split mean accuracy : {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"    Fixed external test accuracy: {fixed_acc:.2%}")
    diff = fixed_acc - mean_acc
    print(f"    Difference                 : {diff:+.2%}")

    # ---- Task 3: Visualise the tree ----
    print(f"\n  --- Task 3: Tree Visualisation ---")
    tree = dt_full.tree_
    print(f"    Depth  : {tree.max_depth}")
    print(f"    Leaves : {tree.n_leaves}")

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(dt_full, ax=ax, filled=True, rounded=True,
              fontsize=5, max_depth=4,
              feature_names=[f"f{i}" for i in range(X_train_full.shape[1])],
              class_names=sorted(set(y_train_full)))
    ax.set_title("Decision Tree (max display depth = 4)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ex2_decision_tree.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {out_dir}/ex2_decision_tree.png")
