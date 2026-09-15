"""Settles plan step 3.7.1b (the optional two-feature variant) and adds the
per-representation confusion analysis that was deferred in the first pass.

Two questions answered here:

1. **3.7.1b - does `(z, t) = (x^2, y^2)` as two separate attributes also give a
   one-node tree?**  Report section 4.3 argued it cannot, because a decision tree
   tests one feature at a time and cannot evaluate the *sum* `z + t`.  This script
   measures the actual tree instead of asserting it, and explicitly searches for
   whether *any* single-feature threshold on this representation could be perfect.

2. **Per-representation confusion.**  The first pass reported accuracy and tree
   size per representation but not *which* points were wrong.  Since the four
   representations produce genuinely different trees, their error patterns should
   differ too - most notably the `sq` variant, whose errors should sit on the
   inner square boundary rather than on the circle.

Outputs
-------
results/representation_twotest.json         3.7.1b evidence
results/representation_confusion.csv        per-representation confusion matrices
results/representation_errors.csv           which grid regions each variant gets wrong
figures/representation_decision_boundaries.png  4-panel decision surface comparison
figures/representation_error_maps.png       2x4 grid of error scatter per variant
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from arff_utils import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    load_arff,
    write_csv,
)
from weka_run import J48, run_weka  # noqa: E402

VARIANTS = ["xy", "sq", "zt", "u"]
TITLES = {
    "xy": "original $(x, y)$",
    "sq": "inner region = square $(x, y)$",
    "zt": "representation $(x^2, y^2)$",
    "u": "representation $u = x^2 + y^2$",
}


def load_variant(split: str, variant: str):
    path = os.path.join(DATA_DIR, f"circle{split}_{variant}.arff")
    X, y_str, attrs, _c, _r = load_arff(path)
    y = np.array([0 if v == "c" else 1 for v in y_str], dtype=int)
    return X, y, attrs


def true_circle_labels(X: np.ndarray) -> np.ndarray:
    """Analytic labels for the original problem, using the x/y coordinates."""
    return np.where(np.einsum("ij,ij->i", X, X) <= 1.0, 0, 1).astype(int)


def grid_ground_truth(X_xy: np.ndarray, file_labels: np.ndarray):
    """Ground truth for the dense grid, plus a boundary mask.

    **The file labels are authoritative**, not a recomputation from the (rounded)
    coordinates.  This matters more than it first appears: 12 grid points have
    ``x^2 + y^2`` equal to 1 in exact decimal arithmetic, and evaluating
    ``x*x + y*y <= 1.0`` on the stored decimal values in binary floating point
    classifies 6 of them as inside and 6 as outside - the opposite split to the
    one the file uses.  Recomputing therefore manufactures phantom "errors" that
    are pure floating-point noise at a measure-zero set.

    The analytic rule is still computed and returned as a cross-check, together
    with a boundary mask identifying the points where the two can legitimately
    disagree.  Those points are excluded from the interior accuracy.

    Returns ``(labels, on_boundary, analytic_labels)``.
    """
    x, y = X_xy[:, 0], X_xy[:, 1]
    r2 = x ** 2 + y ** 2
    analytic = np.where(r2 <= 1.0, 0, 1).astype(int)
    on_boundary = (np.abs(r2 - 1.0) < 1e-12) | (
        np.abs(np.maximum(np.abs(x), np.abs(y)) - 1.25) < 1e-12
    )
    return file_labels.astype(int), on_boundary, analytic


def main() -> None:
    from sklearn.tree import DecisionTreeClassifier, export_text

    out = {}
    confusion_rows = []
    error_rows = []

    # ---------------------------------------------------------------- 3.7.1b
    print("=== 3.7.1b: is (x^2, y^2) as two features a one-node tree? ===")
    X_tr, y_tr, attrs = load_variant("train", "zt")
    X_te, y_te, _ = load_variant("test", "zt")

    # (a) what does the tree actually look like?
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2,
                                 random_state=1).fit(X_tr, y_tr)
    # (b) what is the *minimum* depth needed, ignoring the M=2 constraint?
    clf_full = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1,
                                      ccp_alpha=0.0, random_state=1).fit(X_tr, y_tr)
    print(export_text(clf, feature_names=list(attrs)))
    depth = int(clf.get_depth())
    leaves = int(clf.get_n_leaves())
    print(f"  min_samples_leaf=2 : depth={depth} leaves={leaves} "
          f"train={clf.score(X_tr, y_tr):.4f} test={clf.score(X_te, y_te):.4f}")
    print(f"  fully grown        : depth={clf_full.get_depth()} "
          f"leaves={clf_full.get_n_leaves()} train={clf_full.score(X_tr, y_tr):.4f}")

    # (c) exhaustive search: is there ANY single-feature threshold that is perfect?
    #     i.e. max over c of f < min over q of f, for f in {z, t, z+t, z-t, max, min}
    candidates = {
        "z": X_tr[:, 0],
        "t": X_tr[:, 1],
        "z + t": X_tr[:, 0] + X_tr[:, 1],
        "z - t": X_tr[:, 0] - X_tr[:, 1],
        "max(z, t)": np.maximum(X_tr[:, 0], X_tr[:, 1]),
        "min(z, t)": np.minimum(X_tr[:, 0], X_tr[:, 1]),
    }
    single_feature = {}
    for name, values in candidates.items():
        c_vals, q_vals = values[y_tr == 0], values[y_tr == 1]
        separable = bool(c_vals.max() < q_vals.min())
        single_feature[name] = {
            "max_in_class_c": float(c_vals.max()),
            "min_in_class_q": float(q_vals.min()),
            "separable_by_single_threshold": separable,
        }
        print(f"    f = {name:11s} max|c={c_vals.max():.6f} "
              f"min|q={q_vals.min():.6f} separable={separable}")

    out["zt_two_feature"] = {
        "sklearn_tree_depth_min_samples_leaf_2": depth,
        "sklearn_tree_leaves_min_samples_leaf_2": leaves,
        "sklearn_tree_accuracy_test": float(clf.score(X_te, y_te)),
        "sklearn_tree_fully_grown_leaves": int(clf_full.get_n_leaves()),
        "single_feature_threshold_search": single_feature,
        "conclusion": (
            "Separating z and t gives a tree of depth "
            f"{depth} with {leaves} leaves, NOT a one-node stump: a decision tree "
            "tests one attribute at a time and cannot evaluate the sum z + t. "
            "Only the engineered single feature u = z + t (variant 'u') collapses "
            "the tree to 2 leaves."
        ),
    }
    print(f"  -> {out['zt_two_feature']['conclusion']}")

    # ------------------------------------------------- per-representation errors
    print("\n=== per-representation confusion and error localisation ===")
    X_xy_all, y_xy_all, _ = load_variant("all", "xy")
    y_true_grid, on_boundary, analytic_grid = grid_ground_truth(X_xy_all, y_xy_all)
    interior = ~on_boundary
    print(f"  grid ground truth: {len(y_true_grid)} points (file labels authoritative), "
          f"{int(on_boundary.sum())} on a boundary (excluded from interior accuracy); "
          f"the analytic rule disagrees with the file on "
          f"{int((analytic_grid != y_true_grid).sum())} of them")

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    theta = np.linspace(0, 2 * np.pi, 500)

    for col, variant in enumerate(VARIANTS):
        X_te_v, y_te_v, _ = load_variant("test", variant)
        res = run_weka([J48, "-C", "0.25", "-M", "2"],
                       train=os.path.join(DATA_DIR, f"circletrain_{variant}.arff"),
                       test=os.path.join(DATA_DIR, f"circletest_{variant}.arff"))
        # WEKA's confusion matrix for this variant
        cm = np.array(res.confusion or [], dtype=int)
        labels = res.class_labels
        if cm.size:
            confusion_rows.append({
                "variant": variant, "class_labels": str(labels),
                "matrix": str(cm.tolist()),
                "TP_c": int(cm[0, 0]), "FN_c": int(cm[0, 1]),
                "FP_c": int(cm[1, 0]), "TN_c": int(cm[1, 1]),
                "accuracy_from_matrix": float(np.trace(cm) / cm.sum()),
            })

        # predictions on the dense grid, rendered as a decision surface
        import arff_utils
        from sklearn.tree import DecisionTreeClassifier as DTC

        X_tr_v, y_tr_v, _ = load_variant("train", variant)
        X_all_v, _y_all_v, _ = load_variant("all", variant)
        clf_v = DTC(criterion="entropy", min_samples_leaf=2, random_state=1)
        clf_v.fit(X_tr_v, y_tr_v)
        pred_v = clf_v.predict(X_all_v)
        wrong = pred_v != y_true_grid

        # A tree can only emit a threshold equal to a value seen in training; the
        # largest training u is 0.999033, so a tree on `u` cannot represent the
        # rule `u <= 1` exactly and will misclassify grid points whose u is
        # exactly 1.0 to within floating-point representation.  Those points are
        # already excluded from the headline figure via the boundary mask; they
        # are counted separately here purely so the note in the report is exact.
        boundary_artefact = np.zeros(len(wrong), dtype=bool)
        if variant == "u":
            u_vals = X_all_v[:, 0]
            boundary_artefact = wrong & (np.abs(u_vals - 1.0) < 1e-9) & (pred_v == 1)

        # how many errors lie near the inner-region boundary?
        # For 'sq' the inner boundary is the square max(|x|,|y|)=0.88;
        # for the others it is the circle x^2+y^2=1.
        r = np.sqrt(np.einsum("ij,ij->i", X_xy_all, X_xy_all))
        m = np.maximum(np.abs(X_xy_all[:, 0]), np.abs(X_xy_all[:, 1]))
        if variant == "sq":
            dist_inner = np.abs(m - 0.88)
            inner_name = "square |x|,|y|=0.88"
        else:
            dist_inner = np.abs(r - 1.0)
            inner_name = "circle r=1"
        near = dist_inner < 0.075          # within one grid step
        # Accuracy over the interior only.  Boundary points are *removed from both
        # the numerator and the denominator* - they are not counted as successes.
        # (An earlier version wrote `(~wrong | on_boundary).mean()`, which credits
        # every boundary point as correct and therefore overstates the interior
        # accuracy; that bug is why this is spelled out explicitly.)
        interior_errors = int((wrong & interior).sum())
        interior_accuracy = float((~wrong)[interior].mean())

        error_rows.append({
            "variant": variant,
            "inner_boundary": inner_name,
            "n_grid_errors": int(wrong.sum()),
            "n_points_total": int(len(y_true_grid)),
            "n_points_on_boundary": int(on_boundary.sum()),
            "n_interior_points": int(interior.sum()),
            "n_interior_errors": interior_errors,
            "grid_accuracy": float((~wrong).mean()),
            "grid_accuracy_interior_only": interior_accuracy,
            "n_errors_all_k_equals_1": int(boundary_artefact.sum()),
            "errors_within_1_grid_step_of_inner_boundary": int((wrong & near).sum()),
            "pct_of_errors_near_inner_boundary": float(
                100 * (wrong & near).sum() / max(wrong.sum(), 1)),
            "errors_on_outer_square": int((wrong & (np.abs(m - 1.25) < 0.075)).sum()),
        })
        print(f"  {variant:3s}: errors={int(wrong.sum()):4d}/{len(y_true_grid)} raw "
              f"(acc={float((~wrong).mean()):.4f}); "
              f"interior-only errors={interior_errors:4d}/{int(interior.sum())} "
              f"(acc={interior_accuracy:.4f}); "
              f"{int((wrong & near).sum())} of the raw errors near the {inner_name}")

        # --- decision surface (row 0)
        ax = axes[0, col]
        grid = pred_v.reshape(51, 51)
        ax.pcolormesh(np.linspace(-1.25, 1.25, 51), np.linspace(-1.25, 1.25, 51),
                      grid, cmap="coolwarm", shading="auto", vmin=-0.3, vmax=1.3,
                      alpha=0.85)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5)
        ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="k", lw=1.5,
                               ls="--"))
        if variant == "sq":
            ax.add_patch(Rectangle((-0.88, -0.88), 1.76, 1.76, fill=False,
                                   ec="lime", lw=1.8, ls=":"))
        ax.set_title(f"{TITLES[variant]}\ntest acc={res.accuracy:.1f}%  "
                     f"leaves={res.num_leaves}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
        ax.set_xlabel("x")

        # --- error map (row 1)
        ax = axes[1, col]
        ax.scatter(X_xy_all[~wrong, 0], X_xy_all[~wrong, 1], s=6, c="lightgrey")
        ax.scatter(X_xy_all[wrong, 0], X_xy_all[wrong, 1], s=18, c="red",
                   edgecolors="k", linewidths=0.3,
                   label=f"{int(wrong.sum())} errors")
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.3)
        ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="k",
                               lw=1.3, ls="--"))
        if variant == "sq":
            ax.add_patch(Rectangle((-0.88, -0.88), 1.76, 1.76, fill=False,
                                   ec="lime", lw=1.6, ls=":"))
        ax.set_title(f"errors vs ground truth", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize=8)

    axes[0, 0].set_ylabel("decision surface (red = q, blue = c)")
    axes[1, 0].set_ylabel("errors on the dense grid")
    fig.suptitle("Effect of representation on the decision surface and on where errors occur "
                 "(dash-dot: inner square / circle boundary)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "representation_error_maps.png"), dpi=140)
    plt.close(fig)
    print("  wrote figures/representation_error_maps.png")

    write_csv(os.path.join(RESULTS_DIR, "representation_confusion.csv"),
              confusion_rows,
              ["variant", "class_labels", "matrix", "TP_c", "FN_c", "FP_c", "TN_c",
               "accuracy_from_matrix"])
    write_csv(os.path.join(RESULTS_DIR, "representation_errors.csv"), error_rows,
              ["variant", "inner_boundary", "n_grid_errors", "n_points_total",
               "n_points_on_boundary", "n_interior_points", "n_interior_errors",
               "grid_accuracy", "grid_accuracy_interior_only",
               "n_errors_all_k_equals_1",
               "errors_within_1_grid_step_of_inner_boundary",
               "pct_of_errors_near_inner_boundary", "errors_on_outer_square"])
    out["per_representation_errors"] = error_rows
    out["confusion"] = confusion_rows

    with open(os.path.join(RESULTS_DIR, "representation_twotest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/representation_twotest.json, "
          "results/representation_confusion.csv, results/representation_errors.csv")


if __name__ == "__main__":
    main()
