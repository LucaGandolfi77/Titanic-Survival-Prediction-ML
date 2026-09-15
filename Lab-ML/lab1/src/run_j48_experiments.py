"""Plan sections 3.5-3.7: J48 (WEKA) and sklearn DecisionTree on the four
representations of the circle problem, plus the tree-visualisation figures.

Representations
---------------
* ``xy``  original attributes ``x``, ``y``                    (baseline, p.6)
* ``sq``  inner region replaced by the square, half-side 0.88 (p.7)
* ``zt``  attributes ``z = x^2``, ``t = y^2``                 (p.8)
* ``u``   single attribute ``u = x^2 + y^2``                  (p.9)

Outputs
-------
results/j48_representation.csv     accuracy / tree size for every variant
results/j48_representation.json    WEKA tree text per variant
results/sklearn_representation.csv sklearn counterpart (entropy, no pruning)
figures/tree_*.png                 rendered decision trees
figures/errors_one_node.png        grid errors for the ``u`` representation (p.9 hint)
"""

from __future__ import annotations

import json
import os
import subprocess
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
    LAB1_DIR,
    RESULTS_DIR,
)
from weka_run import J48, run_weka, weka_available  # noqa: E402

VARIANTS = ["xy", "sq", "zt", "u"]
VARIANT_TITLE = {
    "xy": "original $(x,y)$",
    "sq": "inner region = square $(x,y)$",
    "zt": "representation $(x^2, y^2)$",
    "u": "representation $u = x^2+y^2$",
}
SKLEARN_PARAMS = dict(
    criterion="entropy",             # J48 uses information gain
    splitter="best",                 # exact split search
    min_samples_leaf=2,              # matches WEKA -M 2
    random_state=1,                  # matches WEKA -S 1
)




def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    available = weka_available()
    print(f"WEKA available: {available}")
    if not available:
        print("!! WEKA missing - WEKA rows will be recorded as NOT RUN")

    # --------------------------------------------------------------- WEKA
    weka_rows = []
    weka_trees = {}
    for variant in VARIANTS:
        if not available:
            weka_rows.append(
                {
                    "variant": variant, "tool": "weka-J48", "mode": "default(pruned,M=2)",
                    "train_acc": None, "test_acc": None,
                    "num_leaves": None, "size_of_tree": None,
                    "status": "NOT RUN (weka unavailable)",
                }
            )
            continue
        for mode, unpruned, m in (
            ("default(pruned,M=2)", False, 2),
            ("unpruned(-U,M=1)", True, 1),
        ):
            args = [J48, "-C", "0.25", "-M", str(m)]
            if unpruned:
                args.append("-U")
            tr_file = os.path.join(DATA_DIR, f"circletrain_{variant}.arff")
            te_file = os.path.join(DATA_DIR, f"circletest_{variant}.arff")
            res = run_weka(args, train=tr_file, test=te_file)
            # training-set evaluation with exactly the same model
            res_tr = run_weka(args, train=tr_file, test=tr_file,
                              section="Error on training data")
            # the printed tree and its size live in the preamble, not in the
            # evaluation section, so read them separately
            model = run_weka(args, train=tr_file, test=te_file, section="all")
            weka_rows.append(
                {
                    "variant": variant,
                    "tool": "weka-J48",
                    "mode": mode,
                    "train_acc": res_tr.accuracy / 100.0 if res_tr.accuracy is not None else None,
                    "test_acc": res.accuracy / 100.0 if res.accuracy is not None else None,
                    "num_leaves": model.num_leaves,
                    "size_of_tree": model.size_of_tree,
                    "tree_text": model.tree_text,
                    "status": "ok",
                }
            )
            weka_trees[f"{variant}|{mode}"] = {
                "tree_text": model.tree_text,
                "num_leaves": model.num_leaves,
                "size_of_tree": model.size_of_tree,
                "test_accuracy": res.accuracy,
                "train_accuracy": res_tr.accuracy,
                "confusion": res.confusion,
                "class_labels": res.class_labels,
            }
            print(
                f"WEKA J48 {variant:>2} {mode:<20} "
                f"train={res_tr.accuracy if res_tr.accuracy is not None else float('nan'):6.2f}% "
                f"test={res.accuracy if res.accuracy is not None else float('nan'):6.2f}%  "
                f"leaves={model.num_leaves}  size={model.size_of_tree}"
            )

    csv_path = os.path.join(RESULTS_DIR, "j48_representation.csv")
    cols = ["variant", "tool", "mode", "train_acc", "test_acc", "num_leaves",
            "size_of_tree", "status"]
    from arff_utils import write_csv

    write_csv(csv_path, weka_rows, cols)
    print(f"\nwrote {csv_path}")

    with open(os.path.join(RESULTS_DIR, "j48_representation.json"), "w", encoding="utf-8") as fh:
        json.dump(weka_trees, fh, indent=2)
    print("wrote results/j48_representation.json (full tree text per variant)")

    # ------------------------------------------------------------ sklearn
    from sklearn.tree import DecisionTreeClassifier, export_text

    sk_rows = []
    sk_trees = {}
    for variant in VARIANTS:
        X_tr, y_tr, attrs = _load(variant, "train")
        X_te, y_te, _ = _load(variant, "test")
        X_all, y_all, _ = _load(variant, "all")
        for mode, unpruned in (("default(min_samples_leaf=2, ccp=0)", False),
                               ("unpruned(min_samples_leaf=1, ccp=0)", True)):
            clf = DecisionTreeClassifier(
                criterion="entropy",
                min_samples_leaf=1 if unpruned else 2,
                ccp_alpha=0.0,
                random_state=1,
            ).fit(X_tr, y_tr)
            train_acc = float(clf.score(X_tr, y_tr))
            test_acc = float(clf.score(X_te, y_te))
            grid_acc = float(clf.score(X_all, y_all))
            tree_text = export_text(clf, feature_names=list(attrs))
            sk_rows.append(
                {
                    "variant": variant, "tool": "sklearn-DT", "mode": mode,
                    "train_acc": train_acc, "test_acc": test_acc,
                    "num_leaves": int(clf.get_n_leaves()),
                    "size_of_tree": int(clf.tree_.node_count),
                    "grid_acc": grid_acc,
                    "status": "ok",
                }
            )
            sk_trees[f"{variant}|{mode}"] = tree_text
            print(
                f"sklearn  {variant:>2} {mode:<38} train={train_acc:.4f} "
                f"test={test_acc:.4f} leaves={clf.get_n_leaves()} nodes={clf.tree_.node_count}"
            )

    sk_path = os.path.join(RESULTS_DIR, "sklearn_representation.csv")
    cols2 = ["variant", "tool", "mode", "train_acc", "test_acc", "grid_acc",
             "num_leaves", "size_of_tree", "status"]
    with open(sk_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols2) + "\n")
        for r in sk_rows:
            fh.write(",".join(str(r.get(c)) for c in cols2) + "\n")
    print(f"wrote {sk_path}")

    with open(os.path.join(RESULTS_DIR, "sklearn_trees.txt"), "w", encoding="utf-8") as fh:
        for key, text in sk_trees.items():
            fh.write(f"===== {key} =====\n{text}\n\n")

    # ------------------------------------------------- tree figures (sklearn)
    for variant in VARIANTS:
        X_tr, y_tr, _ = _load(variant, "train")
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2,
                                     random_state=1).fit(X_tr, y_tr)
        fig, ax = plt.subplots(figsize=(7, 3.2 if variant != "xy" else 5))
        from sklearn.tree import plot_tree

        plot_tree(
            clf, ax=ax, feature_names=list(_load(variant, "train")[2]),
            class_names=["c (inside)", "q (ring)"], filled=True, impurity=False,
            proportion=False, rounded=True, fontsize=8,
        )
        ax.set_title(f"Decision tree - {VARIANT_TITLE[variant]}  "
                     f"({clf.get_n_leaves()} leaves, {clf.tree_.node_count} nodes)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"tree_{variant}.png"), dpi=150)
        plt.close(fig)
        print(f"wrote figures/tree_{variant}.png")

    # --------------------------------------- grid error map for the u variant
    X_tr, y_tr, _ = _load("u", "train")
    X_all, y_all, _ = _load("u", "all")
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2,
                                 random_state=1).fit(X_tr, y_tr)
    pred = clf.predict(X_all)
    # The u representation keeps only u, so reconstruct (x, y) from the xy file
    # for plotting; row order is identical because the files are generated together.
    X_xy_all, _, _ = _load("xy", "all")
    ok = pred == y_all
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(X_xy_all[ok, 0], X_xy_all[ok, 1], s=10, c="lightgrey", label="correct")
    ax.scatter(X_xy_all[~ok, 0], X_xy_all[~ok, 1], s=26, c="red", edgecolors="k",
               linewidths=0.4, label=f"misclassified ({int((~ok).sum())})")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.6, label="true circle $r=1$")
    ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="k", lw=1.6, ls="--",
                           label="domain $|x|,|y|=1.25$"))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Tree on $u = x^2+y^2$: errors on circleall.arff "
                 f"(accuracy {clf.score(X_all, y_all):.4f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "errors_one_node.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/errors_one_node.png")

    # ------------------------------------------------------ summary json
    summary = {
        "weka_available": available,
        "weka": weka_rows,
        "sklearn": sk_rows,
        "sklearn_hyperparameters_note": (
            "criterion=entropy (J48 uses information gain), min_samples_leaf=2 "
            "(WEKA -M 2), ccp_alpha=0 (no pruning), random_state=1 (WEKA -S 1). "
            "Exact numerical agreement with WEKA is not expected."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "representation_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote results/representation_summary.json")


def _load(variant: str, split: str):
    """Load one generated representation variant."""
    import arff_utils

    path = os.path.join(DATA_DIR, f"circle{split}_{variant}.arff")
    X, y_str, attrs, _classes, _rel = arff_utils.load_arff(path)
    return X, arff_utils.encode_labels(y_str), attrs


if __name__ == "__main__":
    main()
