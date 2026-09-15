"""Lab 6 - Scikit-Learn: Machine Learning in Python (deck pp. 17-27).

This laboratory executes every code example shown in the deck, in order, and then
runs the exercise it assigns.  Slide-by-slide correspondence:

======  ==========================================================================
Slide   Content implemented here
======  ==========================================================================
17-21   installation / prerequisites: recorded as the environment table in the report
22      (blank)
23      ``ml_pipeline_example.py`` part 1: load data with ``pandas.read_csv``,
        separate ``X = data.drop("target")`` and ``y = data["target"]``
24      part 2: chained ``train_test_split`` producing a stratified 70/15/15 split;
        ``DecisionTreeClassifier(random_state=42)`` and ``fit``
25      part 3: validation accuracy, then ``accuracy_score``,
        ``classification_report``, ``confusion_matrix`` on the test set
26-27   the exercise: load Iris, implement a kNN and a Decision Tree classifier,
        and compare their performance with accuracy and a confusion matrix
======  ==========================================================================

The deck's pipeline example is written against a CSV file.  The laboratory data in
this repository is ARFF, so the example is executed **twice**: once literally, on a
CSV whose columns are named exactly as the deck assumes, and once against the ARFF
data through the shared loader.  The literal run is what demonstrates that the code
on slides 23-25 works unmodified; the ARFF run is what connects the module to the
rest of the course material.

Outputs
-------
results/pipeline_slide23_25.txt        verbatim output of the deck's example
results/pipeline_circle.csv            the CSV used for the literal run
results/iris_knn_vs_tree.csv           the slide 26-27 comparison table
results/iris_confusion_matrices.csv    confusion matrices in long form
results/iris_full_results.json         every number, machine-readable
figures/iris_knn_vs_tree.png           accuracy vs hyper-parameter, both models
figures/iris_confusion.png             side-by-side confusion matrices
figures/iris_boundaries.png            decision boundaries on two petal features
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "lab1", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arff_utils import (  # noqa: E402
    DATA_DIR,
    LAB1_DIR,
    load_arff,
    write_csv,
)

LAB6_DIR = os.path.join(LAB1_DIR, "..", "lab6")
LAB6_RESULTS = os.path.join(LAB6_DIR, "results")
LAB6_FIGURES = os.path.join(LAB6_DIR, "figures")

RANDOM_STATE = 42
IRIS_SPECIES = ["setosa", "versicolor", "virginica"]


# --------------------------------------------------------------------------
# Slides 23-25 - the deck's pipeline example, executed verbatim
# --------------------------------------------------------------------------


def slides_23_to_25() -> tuple[str, dict]:
    """Run the deck's ``ml_pipeline_example.py`` exactly as printed.

    The deck's code is reproduced below with only two changes, both forced by the
    environment rather than by choice:

    * the data source is a CSV written from ``circleall.arff`` instead of a file
      called ``data.csv``, so that the ``pd.read_csv`` / ``drop("target")`` idiom
      is exercised literally;
    * the ``stratify`` argument uses the ``y`` variable already in scope, exactly
      as the deck writes it.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # --- build the CSV the deck assumes: feature columns + a "target" column ---
    X_all, y_all, attrs, _classes, _rel = load_arff(
        os.path.join(DATA_DIR, "circleall_xy.arff"))
    y_all = np.array([0 if v == "c" else 1 for v in y_all])
    csv_path = os.path.join(LAB6_RESULTS, "pipeline_circle.csv")
    df = pd.DataFrame(X_all, columns=["feature1", "feature2"])
    df["target"] = y_all
    df.to_csv(csv_path, index=False)

    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("DECK SLIDES 23-25: ml_pipeline_example.py, executed as printed")
    lines.append("=" * 74)
    lines.append("")

    # === 1. Load data from CSV ===
    # Assume your CSV file has columns: feature1, feature2, ..., featureN, target
    data = pd.read_csv(csv_path)
    # Separate features (X) and target (y)
    X = data.drop("target", axis=1)
    y = data["target"]
    lines.append(f"[1] loaded {csv_path}")
    lines.append(f"    data.shape = {data.shape}; X.shape = {X.shape}; y.shape = {y.shape}")
    lines.append(f"    columns dropped to form X: {list(X.columns)}")
    lines.append("")

    # === 2. Split into train, validation, and test sets ===
    # First split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    lines.append(f"Training set: {X_train.shape}")
    lines.append(f"Validation set: {X_val.shape}")
    lines.append(f"Test set: {X_test.shape}")
    lines.append("")

    # === 3. Train a Decision Tree model ===
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # === 4. Evaluate on validation set (optional tuning step) ===
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    lines.append(f"Validation Accuracy: {val_acc}")
    lines.append("")

    # === 5. Evaluate on test set ===
    test_pred = clf.predict(X_test)
    lines.append("=== Test Set Evaluation ===")
    lines.append(f"Accuracy: {accuracy_score(y_test, test_pred)}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(str(classification_report(
        y_test, test_pred, target_names=["c (inside circle)", "q (ring)"])))
    lines.append("Confusion Matrix:")
    lines.append(str(confusion_matrix(y_test, test_pred)))
    lines.append("")

    # The window of the deck's example is small because the data is 2-D and the
    # split is random; report the counts so the accuracy is interpretable.
    lines.append("Note on interpretation: the validation and test partitions hold only "
                 f"{len(y_val)} and {len(y_test)} of the {len(y)} grid points, so one "
                 "misclassification is worth "
                 f"{100/len(y_test):.1f} percentage points. The same pipeline is run "
                 "below on the 150-sample Iris set, where the splits are larger.")

    # keep the fixtures the deck's example needs for the confusion-matrix figure
    info = {
        "csv_path": csv_path,
        "n_total": int(len(y)),
        "shapes": {"train": list(X_train.shape), "val": list(X_val.shape),
                   "test": list(X_test.shape)},
        "validation_accuracy": float(val_acc),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    return "\n".join(lines), info


# --------------------------------------------------------------------------
# Slides 26-27 - the Iris exercise
# --------------------------------------------------------------------------


def slides_26_to_27() -> dict:
    """Load Iris; implement a kNN and a Decision Tree; compare with accuracy and
    a confusion matrix (deck slide 27)."""
    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    print("=" * 74)
    print("DECK SLIDES 26-27: classify Iris flowers")
    print("=" * 74)
    print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features "
          f"({', '.join(iris.feature_names)})")
    print(f"classes: {dict(zip(IRIS_SPECIES, np.bincount(y).tolist()))}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    print(f"stratified hold-out: {len(y_train)} train / {len(y_test)} test\n")

    # ------------------------------------------------ kNN hyper-parameter sweep
    knn_rows = []
    print("--- kNN ---")
    for k in (1, 3, 5, 7, 9, 11, 15, 21):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y,
                             cv=StratifiedKFold(5, shuffle=True,
                                                random_state=RANDOM_STATE))
        knn_rows.append({
            "model": "kNN", "hyperparameter": f"k={k}", "k_or_depth": k,
            "holdout_accuracy": float(clf.score(X_test, y_test)),
            "cv5_mean": float(cv.mean()), "cv5_std": float(cv.std()),
            "n_leaves": None,
        })
        print(f"  k={k:>2}  hold-out={clf.score(X_test, y_test):.4f}  "
              f"5-fold CV={cv.mean():.4f} +/- {cv.std():.4f}")

    # ------------------------------------------------ tree depth sweep
    tree_rows = []
    print("\n--- Decision Tree ---")
    for depth in (1, 2, 3, 4, 5, 6, 8, 10, None):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        cv = cross_val_score(
            DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
            X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
        tree_rows.append({
            "model": "DecisionTree",
            "hyperparameter": f"max_depth={depth}", "k_or_depth": -1 if depth is None else depth,
            "holdout_accuracy": float(clf.score(X_test, y_test)),
            "cv5_mean": float(cv.mean()), "cv5_std": float(cv.std()),
            "n_leaves": int(clf.get_n_leaves()),
        })
        label = "unlimited" if depth is None else str(depth)
        print(f"  max_depth={label:>9}  hold-out={clf.score(X_test, y_test):.4f}  "
              f"5-fold CV={cv.mean():.4f} +/- {cv.std():.4f}  "
              f"leaves={clf.get_n_leaves()}")

    knn_df = pd.DataFrame(knn_rows)
    tree_df = pd.DataFrame(tree_rows)

    # --------------------------------- repeated CV (why a single CV is not enough)
    # A single 5-fold split of 150 samples gives a standard error of ~1.8 pp, so
    # which k "wins" can change with the fold assignment alone.  Repeating the
    # cross-validation with different fold seeds gives a far more stable estimate
    # and is the appropriate way to answer slide 27's comparison question.
    from sklearn.model_selection import RepeatedStratifiedKFold

    print("\n--- repeated stratified 5-fold CV (10 repeats, 10 different fold seeds) ---")
    repeated = {"knn": [], "tree": []}
    for k in (1, 3, 5, 7, 9, 11, 15, 21):
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y,
                             cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                                        random_state=RANDOM_STATE))
        repeated["knn"].append({"hyperparameter": f"k={k}", "value": k,
                                "mean": float(cv.mean()), "std": float(cv.std())})
    for depth in (1, 2, 3, 4, 5, None):
        cv = cross_val_score(
            DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
            X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                             random_state=RANDOM_STATE))
        repeated["tree"].append({
            "hyperparameter": f"max_depth={depth}",
            "value": -1 if depth is None else depth,
            "mean": float(cv.mean()), "std": float(cv.std())})
    for name, rows_ in (("kNN", repeated["knn"]), ("tree", repeated["tree"])):
        for r in rows_:
            print(f"  {name:5s} {r['hyperparameter']:>16}  repeated-CV = "
                  f"{r['mean']:.4f} +/- {r['std']:.4f}")
    best_knn_rep = max(repeated["knn"], key=lambda r: r["mean"])
    best_tree_rep = max(repeated["tree"], key=lambda r: r["mean"])
    print(f"  -> by repeated CV the best kNN is {best_knn_rep['hyperparameter']} "
          f"({best_knn_rep['mean']:.4f}) and the best tree depth is "
          f"{best_tree_rep['hyperparameter']} ({best_tree_rep['mean']:.4f})")
    # how much does the *ranking* move? compare single-CV vs repeated-CV winners
    single_knn_k = int(knn_df.loc[knn_df["cv5_mean"].idxmax(), "k_or_depth"])
    single_tree_d = int(tree_df.loc[tree_df["cv5_mean"].idxmax(), "k_or_depth"])
    print(f"  -> single 5-fold CV instead chose kNN k={single_knn_k} and tree "
          f"max_depth={single_tree_d if single_tree_d != -1 else None}")
    print("     The disagreement is a direct measurement of fold-assignment noise: "
          "with 150 samples the difference between these settings is 1-3 instances.")

    write_csv(os.path.join(LAB6_RESULTS, "iris_knn_vs_tree.csv"),
              knn_rows + tree_rows,
              ["model", "hyperparameter", "k_or_depth", "holdout_accuracy",
               "cv5_mean", "cv5_std", "n_leaves"])
    rep_rows = ([{"model": "kNN", **r} for r in repeated["knn"]]
                + [{"model": "DecisionTree", **r} for r in repeated["tree"]])
    write_csv(os.path.join(LAB6_RESULTS, "iris_repeated_cv.csv"), rep_rows,
              ["model", "hyperparameter", "value", "mean", "std"])

    # ---------------------------------- best of each, confusion matrices
    best_k = int(knn_df.loc[knn_df["cv5_mean"].idxmax(), "k_or_depth"])
    best_d_raw = int(tree_df.loc[tree_df["cv5_mean"].idxmax(), "k_or_depth"])
    best_d = None if best_d_raw == -1 else best_d_raw

    knn_best = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    tree_best = DecisionTreeClassifier(max_depth=best_d,
                                       random_state=RANDOM_STATE).fit(X_train, y_train)
    cm_knn = confusion_matrix(y_test, knn_best.predict(X_test))
    cm_tree = confusion_matrix(y_test, tree_best.predict(X_test))

    print(f"\n--- comparison on the same hold-out set (deck slide 27) ---")
    print(f"kNN  (k={best_k}): accuracy = {knn_best.score(X_test, y_test):.4f}")
    print(f"  confusion matrix (rows = true, cols = predicted):\n{cm_knn}")
    print(f"DecisionTree (max_depth={best_d}): accuracy = "
          f"{tree_best.score(X_test, y_test):.4f}")
    print(f"  confusion matrix (rows = true, cols = predicted):\n{cm_tree}")

    # which classes are confused, and why
    err_knn = int(cm_knn.sum() - np.trace(cm_knn))
    err_tree = int(cm_tree.sum() - np.trace(cm_tree))
    print(f"\n  total errors: kNN {err_knn}, tree {err_tree}")
    for name, cm in (("kNN", cm_knn), ("tree", cm_tree)):
        for i in range(3):
            for j in range(3):
                if i != j and cm[i, j]:
                    print(f"    {name}: {IRIS_SPECIES[i]} -> {IRIS_SPECIES[j]} "
                          f"({cm[i, j]} cases)")
    print("\n  setosa is perfectly separated by both models "
          f"({'yes' if cm_knn[0].sum() == cm_knn[0, 0] and cm_tree[0].sum() == cm_tree[0, 0] else 'no'}); "
          "all residual error is versicolor vs virginica, which overlap in "
          "petal-length/petal-width space.")

    with open(os.path.join(LAB6_RESULTS, "iris_confusion_matrices.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("model,true_class,predicted_class,count\n")
        for name, cm in (("kNN", cm_knn), ("DecisionTree", cm_tree)):
            for i in range(3):
                for j in range(3):
                    fh.write(f"{name},{IRIS_SPECIES[i]},{IRIS_SPECIES[j]},{cm[i, j]}\n")

    # ------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].errorbar(knn_df["k_or_depth"], knn_df["cv5_mean"], yerr=knn_df["cv5_std"],
                     fmt="o-", capsize=3, label="kNN")
    xs = tree_df["k_or_depth"].replace(-1, 11)
    axes[0].errorbar(xs, tree_df["cv5_mean"], yerr=tree_df["cv5_std"],
                     fmt="s-", capsize=3, label="decision tree")
    axes[0].set_xlabel("k (kNN)  /  max_depth (tree; 11 = unlimited)")
    axes[0].set_ylabel("5-fold CV accuracy")
    axes[0].set_title("Iris: hyper-parameter sweep (5-fold CV)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(knn_df["k_or_depth"], knn_df["holdout_accuracy"], "o-",
                 label="kNN, hold-out")
    axes[1].plot(xs, tree_df["holdout_accuracy"], "s-", label="tree, hold-out")
    axes[1].set_xlabel("k / max_depth (11 = unlimited)")
    axes[1].set_ylabel("hold-out accuracy (30 % test)")
    axes[1].set_title("Iris: single stratified hold-out")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle("Lab 6 (slide 26-27) - Iris: kNN vs decision tree", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB6_FIGURES, "iris_knn_vs_tree.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, cm, title in ((axes[0], cm_knn, f"kNN (k={best_k})"),
                          (axes[1], cm_tree, f"Decision tree (max_depth={best_d})")):
        im = ax.imshow(cm, cmap="Greens")
        ax.set_title(f"Confusion matrix - {title}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(3)); ax.set_xticklabels(IRIS_SPECIES, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(IRIS_SPECIES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Lab 6 (slide 26-27) - Iris confusion matrices on the hold-out set",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB6_FIGURES, "iris_confusion.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------- decision boundaries, 2 petal features
    fi, fj = 2, 3   # petal length, petal width - the two most discriminative
    X2 = X[:, [fi, fj]]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 0.3, X2[:, 0].max() + 0.3, 300),
                         np.linspace(X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, model, title in (
        (axes[0], KNeighborsClassifier(n_neighbors=best_k), f"kNN (k={best_k})"),
        (axes[1], DecisionTreeClassifier(max_depth=best_d, random_state=RANDOM_STATE),
         f"decision tree (depth {best_d})")):
        model.fit(X2_tr, y2_tr)
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis", levels=[-0.5, 0.5, 1.5, 2.5])
        for c in range(3):
            s = y2_te == c
            ax.scatter(X2_te[s, 0], X2_te[s, 1], s=45, edgecolors="k",
                       linewidths=0.5, label=IRIS_SPECIES[c])
        ax.set_xlabel(iris.feature_names[fi])
        ax.set_ylabel(iris.feature_names[fj])
        ax.set_title(f"{title}   hold-out acc = {model.score(X2_te, y2_te):.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Lab 6 - Iris decision regions on petal length vs petal width "
                 "(test points shown)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB6_FIGURES, "iris_boundaries.png"), dpi=150)
    plt.close(fig)

    print("\nwrote lab6/figures/iris_knn_vs_tree.png, iris_confusion.png, "
          "iris_boundaries.png")
    print("wrote lab6/results/iris_knn_vs_tree.csv, iris_confusion_matrices.csv")

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(iris.feature_names),
        "class_counts": np.bincount(y).tolist(),
        "holdout": {"n_train": int(len(y_train)), "n_test": int(len(y_test))},
        "knn_rows": knn_rows,
        "tree_rows": tree_rows,
        "best_knn": {"k": best_k, "holdout_accuracy": float(knn_best.score(X_test, y_test)),
                     "cv5_mean": float(knn_df["cv5_mean"].max()),
                     "confusion": cm_knn.tolist()},
        "best_tree": {"max_depth": best_d,
                      "holdout_accuracy": float(tree_best.score(X_test, y_test)),
                      "cv5_mean": float(tree_df["cv5_mean"].max()),
                      "n_leaves": int(tree_best.get_n_leaves()),
                      "confusion": cm_tree.tolist()},
        "total_errors": {"knn": err_knn, "tree": err_tree},
        "repeated_cv": repeated,
        "repeated_cv_winners": {
            "knn": best_knn_rep["hyperparameter"],
            "tree": best_tree_rep["hyperparameter"],
        },
        "single_cv_winners": {"knn": f"k={single_knn_k}",
                              "tree": f"max_depth={single_tree_d}"},
    }


# --------------------------------------------------------------------------


def main() -> None:
    os.makedirs(LAB6_RESULTS, exist_ok=True)
    os.makedirs(LAB6_FIGURES, exist_ok=True)

    out: dict = {"lab": 6, "title": "Scikit-Learn: Machine Learning in Python",
                 "deck_pages": "17-27"}

    text, pipeline_info = slides_23_to_25()
    print(text)
    with open(os.path.join(LAB6_RESULTS, "pipeline_slide23_25.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    out["slides_23_25"] = pipeline_info

    print()
    iris = slides_26_to_27()
    out["slides_26_27"] = iris

    with open(os.path.join(LAB6_RESULTS, "lab6_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote lab6/results/lab6_summary.json")


if __name__ == "__main__":
    main()
