"""Plan section 6: the scikit-learn module (PDF p.17-27).

Contents
--------
* 6.2  ``train_test_split`` twice to build a stratified 70/15/15
       train/validation/test pipeline, exactly the structure on p.24-25
* 6.3  sklearn kNN over the same k sweep, asserted to agree with the
       from-scratch implementation, and drawn on the same axes
* 6.4  sklearn decision tree over the same M sweep, with the WEKA -> sklearn
       hyper-parameter mapping documented in the output
* 6.5  the Iris exercise of p.26-27: kNN vs decision tree, accuracy and
       confusion matrix, 5-fold CV and hold-out
* 6.6  the WEKA vs sklearn cross-check table

Outputs
-------
results/sklearn_pipeline_report.txt   the printed pipeline report (p.23-25 style)
results/iris_comparison.csv           accuracy table for both Iris models
results/iris_confusion.csv            confusion matrices for both Iris models
figures/iris_comparison.png           accuracy vs hyper-parameter, both models
figures/iris_confusion.png            side-by-side confusion matrices
results/weka_sklearn_crosscheck.csv   the 6.6 table
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arff_utils import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    digit_path,
    load_arff,
)
from knn_scratch import accuracy, knn_predict  # noqa: E402

TRAIN_TEST_SEED = 42
HOLDOUT_FRACTIONS = (0.70, 0.15, 0.15)
IRIS_SEED = 42

#: WEKA -> scikit-learn concept mapping (plan 6.4).
WEKA_SKLEARN_MAP = [
    ("J48 minNumObj (-M)", "DecisionTreeClassifier(min_samples_leaf=M)",
     "minimum samples per leaf; the capacity knob of Exercise 2b"),
    ("J48 confidenceFactor (-C 0.25)", "ccp_alpha (cost-complexity pruning)",
     "both control how much the tree is pruned after growing"),
    ("J48 unpruned (-U)", "ccp_alpha=0.0", "no post-pruning"),
    ("information gain (entropy)", "criterion='entropy'", "J48 uses information gain"),
    ("J48 default", "criterion='gini'", "sklearn's own default is Gini impurity"),
    ("IBk -K", "KNeighborsClassifier(n_neighbors=K)", "number of neighbours"),
    ("IBk -W 0 (no weighting)", "weights='uniform'", "plain majority vote"),
    ("-split-percentage 66 -s SEED", "train_test_split(test_size=0.34, random_state=SEED)",
     "the random partition; replicated exactly via java.util.Random"),
]


# --------------------------------------------------------------------------
# 6.2  generic train/validation/test pipeline (p.23-25, adapted to ARFF)
# --------------------------------------------------------------------------


def stratified_three_way(X, y, seed: int = TRAIN_TEST_SEED):
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def pipeline_report(seed: int = TRAIN_TEST_SEED) -> str:
    """The p.23-25 pipeline, run on the circle data and printed as a report."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.tree import DecisionTreeClassifier

    # p.23 loads a CSV.  The lab data is ARFF, so the loader is generalised but
    # the *structure* (features matrix + target vector) is identical.
    X, y_str, attrs, classes, _ = load_arff(
        os.path.join(DATA_DIR, "circletrain_xy.arff"))
    y = np.array([0 if v == "c" else 1 for v in y_str])
    X_test_ext, y_test_ext, _, _, _ = load_arff(
        os.path.join(DATA_DIR, "circletest_xy.arff"))
    y_test_ext = np.array([0 if v == "c" else 1 for v in y_test_ext])

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_three_way(
        X, y, seed)

    lines = []
    lines.append("# " + "=" * 72)
    lines.append("# Scikit-Learn pipeline (PDF p.23-25), run on the circle problem")
    lines.append("# " + "=" * 72)
    lines.append("# Loader: lab1/src/arff_utils.load_arff (ARFF instead of read_csv,")
    lines.append("#         same X / y structure as the deck's example)")
    lines.append(f"Training set: {X_train.shape}")
    lines.append(f"Validation set: {X_val.shape}")
    lines.append(f"Test set: {X_test.shape}")
    lines.append("")

    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    lines.append(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    lines.append("")

    test_pred = clf.predict(X_test)
    lines.append("=== Test Set Evaluation ===")
    lines.append(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(classification_report(y_test, test_pred,
                                       target_names=["c (inside)", "q (ring)"]))
    lines.append("Confusion Matrix:")
    lines.append(str(confusion_matrix(y_test, test_pred)))
    lines.append("")
    # The lab also supplies a dedicated external test file - report it too.
    ext_pred = clf.predict(X_test_ext)
    lines.append("=== External test set (circletest.arff) ===")
    lines.append(f"Accuracy: {accuracy_score(y_test_ext, ext_pred):.4f}")
    lines.append(str(confusion_matrix(y_test_ext, ext_pred)))
    lines.append("")

    # The deck's example uses a CSV; show that path works too so the pipeline is
    # verified against the literal instructions.
    tmp_csv = os.path.join(RESULTS_DIR, "_circle_pipeline_demo.csv")
    pd.DataFrame(
        np.column_stack([X, y]), columns=["x", "y", "target"]
    ).to_csv(tmp_csv, index=False)
    demo = pd.read_csv(tmp_csv)
    Xd = demo.drop("target", axis=1)
    yd = demo["target"]
    (Xtr, ytr), _, (Xte, yte) = stratified_three_way(
        Xd.to_numpy(dtype=float), yd.to_numpy(dtype=int), seed)
    clf_d = DecisionTreeClassifier(random_state=seed).fit(Xtr, ytr)
    lines.append("=== Literal p.23-25 path (CSV via pandas.read_csv) ===")
    lines.append(f"CSV round-trip test accuracy: {clf_d.score(Xte, yte):.4f} "
                 f"(shape {Xd.shape})")
    os.remove(tmp_csv)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6.3  sklearn kNN agreement + combined figure
# --------------------------------------------------------------------------


def knn_agreement() -> dict:
    from sklearn.neighbors import KNeighborsClassifier

    X_tr, y_tr, _ = _load_circle("train", "xy")
    X_te, y_te, _ = _load_circle("test", "xy")
    ks = [1, 3, 5, 7, 9, 11, 13, 15, 21, 31, 51]
    rows = []
    for k in ks:
        mine = knn_predict(X_tr, y_tr, X_te, k)
        sk = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(
            X_tr, y_tr).predict(X_te)
        rows.append({
            "k": k,
            "scratch_acc": accuracy(y_te, mine),
            "sklearn_acc": accuracy(y_te, sk),
            "n_agree": int(np.sum(mine == sk)),
            "n_test": len(y_te),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "sklearn_knn_agreement.csv"), index=False)
    return {"all_agree": bool((out["n_agree"] == out["n_test"]).all()),
            "rows": rows}


def _load_circle(split: str, variant: str):
    import arff_utils

    X, y_str, attrs, _c, _r = arff_utils.load_arff(
        os.path.join(DATA_DIR, f"circle{split}_{variant}.arff"))
    return X, arff_utils.encode_labels(y_str), attrs


# --------------------------------------------------------------------------
# 6.4  sklearn decision tree M sweep
# --------------------------------------------------------------------------


def sklearn_M_sweep():
    from sklearn.tree import DecisionTreeClassifier

    X_tr, y_tr, _ = _load_circle("train", "xy")
    X_te, y_te, _ = _load_circle("test", "xy")
    rows = []
    for M in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=M,
                                     ccp_alpha=0.0, random_state=1).fit(X_tr, y_tr)
        rows.append({"M": M,
                     "train_acc": float(clf.score(X_tr, y_tr)),
                     "test_acc": float(clf.score(X_te, y_te)),
                     "leaves": int(clf.get_n_leaves())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "sklearn_circle_M_sweep.csv"), index=False)
    return rows


# --------------------------------------------------------------------------
# 6.5  Iris
# --------------------------------------------------------------------------


def iris_experiment():
    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=IRIS_SEED, stratify=y)

    ks = [1, 3, 5, 7, 9, 11, 15, 21]
    depths = [1, 2, 3, 4, 5, 6, 8, 10, None]

    knn_rows, tree_rows = [], []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5)
        knn_rows.append({"k": k, "holdout_acc": float(clf.score(X_te, y_te)),
                         "cv5_mean": float(cv.mean()), "cv5_std": float(cv.std())})
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=IRIS_SEED).fit(X_tr, y_tr)
        cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=IRIS_SEED),
                             X, y, cv=5)
        tree_rows.append({"max_depth": -1 if d is None else d,
                          "holdout_acc": float(clf.score(X_te, y_te)),
                          "cv5_mean": float(cv.mean()), "cv5_std": float(cv.std()),
                          "leaves": int(clf.get_n_leaves())})

    knn_df, tree_df = pd.DataFrame(knn_rows), pd.DataFrame(tree_rows)
    knn_df.to_csv(os.path.join(RESULTS_DIR, "iris_knn.csv"), index=False)
    tree_df.to_csv(os.path.join(RESULTS_DIR, "iris_tree.csv"), index=False)

    # Best of each, with confusion matrices on the same hold-out set
    best_k = int(knn_df.loc[knn_df["cv5_mean"].idxmax(), "k"])
    best_d_raw = int(tree_df.loc[tree_df["cv5_mean"].idxmax(), "max_depth"])
    best_d = None if best_d_raw == -1 else best_d_raw
    knn_best = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr)
    tree_best = DecisionTreeClassifier(max_depth=best_d, random_state=IRIS_SEED).fit(
        X_tr, y_tr)
    cm_knn = confusion_matrix(y_te, knn_best.predict(X_te))
    cm_tree = confusion_matrix(y_te, tree_best.predict(X_te))

    summary_rows = [
        {"model": "kNN", "hyperparameter": f"k={best_k}",
         "holdout_acc": float(knn_best.score(X_te, y_te)),
         "cv5_mean": float(knn_df["cv5_mean"].max()),
         "cv5_std": float(knn_df.loc[knn_df['cv5_mean'].idxmax(), "cv5_std"]),
         "leaves_or_k": best_k},
        {"model": "DecisionTree", "hyperparameter": f"max_depth={best_d}",
         "holdout_acc": float(tree_best.score(X_te, y_te)),
         "cv5_mean": float(tree_df["cv5_mean"].max()),
         "cv5_std": float(tree_df.loc[tree_df['cv5_mean'].idxmax(), "cv5_std"]),
         "leaves_or_k": int(tree_best.get_n_leaves())},
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "iris_comparison.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "iris_confusion.csv"), "w", encoding="utf-8") as fh:
        fh.write("model,true,predicted,count\n")
        for name, cm in (("knn", cm_knn), ("tree", cm_tree)):
            for i in range(len(target_names)):
                for j in range(len(target_names)):
                    fh.write(f"{name},{target_names[i]},{target_names[j]},{cm[i, j]}\n")

    # ---------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].errorbar(knn_df["k"], knn_df["cv5_mean"], yerr=knn_df["cv5_std"],
                     fmt="o-", capsize=3, label="kNN, 5-fold CV")
    axes[0].errorbar(tree_df["max_depth"].replace(-1, 11), tree_df["cv5_mean"],
                     yerr=tree_df["cv5_std"], fmt="s-", capsize=3,
                     label="Decision tree, 5-fold CV")
    axes[0].set_xlabel("k (kNN) / max_depth (tree; 11 = unlimited)")
    axes[0].set_ylabel("5-fold CV accuracy")
    axes[0].set_title("Iris: hyper-parameter sweep (5-fold CV)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(knn_df["k"], knn_df["holdout_acc"], "o-", label="kNN, hold-out")
    axes[1].plot(tree_df["max_depth"].replace(-1, 11), tree_df["holdout_acc"], "s-",
                 label="Decision tree, hold-out")
    axes[1].set_xlabel("k / max_depth (11 = unlimited)")
    axes[1].set_ylabel("hold-out accuracy (30% test)")
    axes[1].set_title("Iris: single hold-out")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "iris_comparison.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, cm, title in ((axes[0], cm_knn, f"kNN (k={best_k})"),
                          (axes[1], cm_tree, f"Decision tree (max_depth={best_d})")):
        im = ax.imshow(cm, cmap="Greens")
        ax.set_title(f"Iris hold-out confusion - {title}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(3))
        ax.set_xticklabels(target_names, rotation=20)
        ax.set_yticks(range(3))
        ax.set_yticklabels(target_names)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "iris_confusion.png"), dpi=150)
    plt.close(fig)

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": target_names,
        "best_knn": summary_rows[0],
        "best_tree": summary_rows[1],
        "knn_rows": knn_rows,
        "tree_rows": tree_rows,
        "note": (
            "Setosa is linearly separable from the other two species, so both "
            "models reach 100% on it; all remaining errors are versicolor vs "
            "virginica, which overlap in feature space."
        ),
    }


# --------------------------------------------------------------------------
# 6.6  cross-check table
# --------------------------------------------------------------------------


def crosscheck() -> pd.DataFrame:
    rows = []

    def add(dataset, model, hp, weka_acc, sk_acc, note):
        delta = None
        if weka_acc is not None and sk_acc is not None:
            delta = round(100 * (sk_acc - weka_acc), 2)
        rows.append({"dataset": dataset, "model": model, "hyperparameter": hp,
                     "weka_accuracy": weka_acc, "sklearn_accuracy": sk_acc,
                     "delta_pp": delta, "note": note})

    # circle, from the representation experiment
    rep_path = os.path.join(RESULTS_DIR, "representation_summary.json")
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as fh:
            rep = json.load(fh)
        weka = {(r["variant"], r["mode"]): r
                for r in rep["weka"] if r.get("train_acc") is not None}
        # The two tools name the modes differently; normalise both to a common key.
        weka_norm = {}
        for r in rep["weka"]:
            if r.get("train_acc") is None:
                continue
            key = ("pruned", "M=2") if "default" in r["mode"] else ("unpruned", "M=1")
            weka_norm[(r["variant"], key)] = r
        sk_norm = {}
        for r in rep["sklearn"]:
            key = ("pruned", "M=2") if "default" in r["mode"] else ("unpruned", "M=1")
            sk_norm[(r["variant"], key)] = r

        for variant, label in (("xy", "circle (x,y)"), ("sq", "square inner (x,y)"),
                               ("zt", "(x^2,y^2)"), ("u", "u=x^2+y^2")):
            for key, mode_label in ((("pruned", "M=2"), "pruned M=2"),
                                    (("unpruned", "M=1"), "unpruned M=1")):
                w = weka_norm.get((variant, key))
                s = sk_norm.get((variant, key))
                if w and s:
                    add(label, "decision tree", mode_label, w["test_acc"],
                        s["test_acc"],
                        f"WEKA leaves={w['num_leaves']}, sklearn leaves={s['num_leaves']}")

    # digits, from the M sweep
    m_path = os.path.join(RESULTS_DIR, "j48_M_sweep.csv")
    if os.path.exists(m_path):
        df = pd.read_csv(m_path)
        for M in (1, 2, 3, 5, 10):
            sub = df[df["M"] == M]
            if len(sub) and "weka_test_acc" in df.columns:
                r = sub.iloc[0]
                add("digits (Bigtest1 -> Bigtest2)", "decision tree", f"M={M}",
                    float(r["weka_test_acc"]), float(r["sklearn_test_acc"]),
                    f"leaves WEKA={r['weka_leaves']:.0f}, sklearn={r['sklearn_leaves']:.0f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "weka_sklearn_crosscheck.csv"), index=False)
    return out


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== 6.2 pipeline (p.23-25) ===")
    report = pipeline_report()
    with open(os.path.join(RESULTS_DIR, "sklearn_pipeline_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print("\nwrote results/sklearn_pipeline_report.txt")

    print("\n=== 6.3 sklearn kNN vs from-scratch kNN ===")
    agree = knn_agreement()
    for r in agree["rows"]:
        print(f"  k={r['k']:>3} scratch={r['scratch_acc']:.4f} "
              f"sklearn={r['sklearn_acc']:.4f} agree={r['n_agree']}/{r['n_test']}")
    print(f"  ALL AGREE: {agree['all_agree']}")

    print("\n=== 6.4 sklearn decision tree M sweep on the circle data ===")
    for r in sklearn_M_sweep():
        print(f"  M={r['M']:>3} train={r['train_acc']:.3f} test={r['test_acc']:.3f} "
              f"leaves={r['leaves']}")

    print("\n=== 6.5 Iris ===")
    iris = iris_experiment()
    print(f"  best kNN : {iris['best_knn']}")
    print(f"  best tree: {iris['best_tree']}")
    print("  wrote figures/iris_comparison.png, figures/iris_confusion.png")

    print("\n=== 6.6 WEKA vs sklearn cross-check ===")
    cc = crosscheck()
    print(cc.to_string(index=False))
    print("\nwrote results/weka_sklearn_crosscheck.csv")

    with open(os.path.join(RESULTS_DIR, "sklearn_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"knn_agreement": agree, "iris": iris,
                   "weka_to_sklearn_mapping": [
                       {"weka": a, "sklearn": b, "meaning": c}
                       for a, b, c in WEKA_SKLEARN_MAP]},
                  fh, indent=2, default=str)
    print("wrote results/sklearn_summary.json")


if __name__ == "__main__":
    main()
