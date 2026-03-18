"""exercise1b_repr.py — Feature engineering variants for the circle/square problem.

Part A: Relabel data as inner-square vs outer-region; train Decision Tree.
Part B: Quadratic representation (z=x², t=y²) for both problems.
Part C: Optimal single-feature r²=x²+y²; achieves ~100 % on circle problem.

Outputs:
  - Printed accuracy tables and tree structures
  - outputs/ex1b_optimal_repr.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text

from data_loader import load_arff, arff_features_and_label, write_arff
from utils import print_metrics


# ------------------------------------------------------------------
# Part A — Inner Square Relabeling
# ------------------------------------------------------------------

def _relabel_inner_square(df: pd.DataFrame, half_side: float = 0.88) -> pd.DataFrame:
    """Relabel: 'c' if inside axis-aligned inner square, 'q' otherwise."""
    df = df.copy()
    xs = df.iloc[:, 0].values.astype(float)
    ys = df.iloc[:, 1].values.astype(float)
    labels = np.where(
        (np.abs(xs) <= half_side) & (np.abs(ys) <= half_side), "c", "q"
    )
    df["label"] = labels
    return df


def part_a(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n  --- Part A: Inner-Square Relabeling ---")

    df_train = load_arff(f"{data_dir}/circletrain.arff")
    df_test = load_arff(f"{data_dir}/circletest.arff")

    df_train_sq = _relabel_inner_square(df_train)
    df_test_sq = _relabel_inner_square(df_test)

    # Save relabeled ARFF files
    write_arff(df_train_sq, "circle_square", f"{data_dir}/circletrain_square.arff")
    write_arff(df_test_sq, "circle_square", f"{data_dir}/circletest_square.arff")
    print(f"  Saved relabeled data to {data_dir}/circletrain_square.arff "
          f"and circletest_square.arff")

    X_train, y_train = arff_features_and_label(df_train_sq)
    X_test, y_test = arff_features_and_label(df_test_sq)

    dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    pred = dt.predict(X_test)
    acc = print_metrics(y_test, pred, "DecTree inner-square")

    print(
        "\n  Explanation:"
        "\n  The inner-square boundary is defined by axis-aligned conditions:"
        "\n    |x| ≤ 0.88  AND  |y| ≤ 0.88"
        "\n  Decision Trees split on individual features with threshold tests,"
        "\n  so they can represent this boundary exactly (e.g. x ≤ 0.88 and"
        "\n  x ≥ -0.88, etc.). This makes the square problem ideally suited"
        "\n  for axis-aligned Decision Trees."
    )


# ------------------------------------------------------------------
# Part B — Quadratic Representation  z=x², t=y²
# ------------------------------------------------------------------

def _quadratic_transform(X: np.ndarray) -> np.ndarray:
    """Return (z, t) = (x², y²)."""
    return X ** 2


def part_b(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n  --- Part B: Quadratic Representation (z=x², t=y²) ---")

    results: list[dict] = []

    for problem, relabel_fn in [("circle", None), ("inner-square", _relabel_inner_square)]:
        df_train = load_arff(f"{data_dir}/circletrain.arff")
        df_test = load_arff(f"{data_dir}/circletest.arff")

        if relabel_fn is not None:
            df_train = relabel_fn(df_train)
            df_test = relabel_fn(df_test)

        X_train, y_train = arff_features_and_label(df_train)
        X_test, y_test = arff_features_and_label(df_test)

        # Original features
        dt_orig = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        acc_orig = float(np.mean(dt_orig.predict(X_test) == y_test))

        # Quadratic features
        Xq_train = _quadratic_transform(X_train)
        Xq_test = _quadratic_transform(X_test)
        dt_quad = DecisionTreeClassifier(random_state=42).fit(Xq_train, y_train)
        acc_quad = float(np.mean(dt_quad.predict(Xq_test) == y_test))

        results.append({
            "problem": problem,
            "acc_original": acc_orig,
            "acc_quadratic": acc_quad,
        })

    print(f"\n  {'Problem':<16} {'Original':>10} {'Quadratic':>10}")
    print("  " + "-" * 38)
    for r in results:
        print(f"  {r['problem']:<16} {r['acc_original']:>10.2%} "
              f"{r['acc_quadratic']:>10.2%}")

    print(
        "\n  Note: In the quadratic representation, the circle boundary"
        "\n  x²+y² ≤ 1 becomes z+t ≤ 1, which can be approximated by"
        "\n  axis-aligned splits much more easily than the original circle."
    )


# ------------------------------------------------------------------
# Part C — Optimal Single Feature  r² = x²+y²
# ------------------------------------------------------------------

def part_c(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n  --- Part C: Optimal Feature r² = x²+y² ---")

    df_train = load_arff(f"{data_dir}/circletrain.arff")
    df_test = load_arff(f"{data_dir}/circletest.arff")
    df_all = load_arff(f"{data_dir}/circleall.arff")

    X_train, y_train = arff_features_and_label(df_train)
    X_test, y_test = arff_features_and_label(df_test)
    X_all, y_all = arff_features_and_label(df_all)

    # Single feature: r²
    r2_train = (X_train ** 2).sum(axis=1, keepdims=True)
    r2_test = (X_test ** 2).sum(axis=1, keepdims=True)
    r2_all = (X_all ** 2).sum(axis=1, keepdims=True)

    dt = DecisionTreeClassifier(random_state=42).fit(r2_train, y_train)

    acc_test = float(np.mean(dt.predict(r2_test) == y_test))
    acc_all = float(np.mean(dt.predict(r2_all) == y_all))

    print(f"  Test accuracy : {acc_test:.2%}")
    print(f"  All accuracy  : {acc_all:.2%}")

    # Print tree structure
    print("\n  Tree structure:")
    tree_text = export_text(dt, feature_names=["r_squared"])
    for line in tree_text.split("\n"):
        print(f"    {line}")

    # Plot circleall predicted labels
    pred_all = dt.predict(r2_all)
    label_map = {"c": 0, "q": 1}
    c_pred = [label_map.get(l, 0) for l in pred_all]
    c_true = [label_map.get(l, 0) for l in y_all]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(X_all[:, 0], X_all[:, 1], c=c_true, cmap="coolwarm",
                    s=4, alpha=0.6)
    axes[0].set_title("True Labels")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    axes[1].scatter(X_all[:, 0], X_all[:, 1], c=c_pred, cmap="coolwarm",
                    s=4, alpha=0.6)
    axes[1].set_title(f"Predicted (r² tree, acc={acc_all:.2%})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    fig.suptitle("Optimal Representation: r² = x²+y²", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ex1b_optimal_repr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved: {out_dir}/ex1b_optimal_repr.png")


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run(data_dir: str = "data", out_dir: str = "outputs") -> None:
    print("\n" + "=" * 64)
    print("Exercise 1b — Feature Engineering Variants")
    print("=" * 64)

    part_a(data_dir, out_dir)
    part_b(data_dir, out_dir)
    part_c(data_dir, out_dir)
