"""Lab 2 - Clustering.  K-Means and X-Means on the gaussian and digit data.

Covers the exercises on pages 28-43 of the merged deck:

* 2.1  K-Means on ``gausstrain.arff``, "Use training set" mode (ignore the class)
* 2.2  Where the centroids land, and how far they are from the true Gaussian means
* 2.3  "Classes to clusters evaluation" - confusion matrix and accuracy
* 2.4  ``gausstrainhv.arff``: the same problem with 50% more variance
* 2.5  K-Means with 10 clusters on the digit images; centroids rendered as greyscale
* 2.6  X-Means-equivalent: how many clusters does the data support, and overfitting

WEKA's ``SimpleKMeans`` uses k-means++ initialisation (``-init 0``) and Euclidean
distance.  scikit-learn's ``KMeans(n_init=10)`` is used as the reference
implementation; the two agree up to label permutation.

Note on the class attribute: the gaussian files declare ``cluster {1,2}``, a
ground-truth grouping that the clustering is *not* allowed to see.  It is used
only for the "classes to clusters" evaluation of exercise 2.3.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The ARFF reader/writer and CSV helper are shared with Lab 1.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "lab1", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arff_utils import (  # noqa: E402
    FIGURES_DIR,
    LAB1_DIR,
    RESULTS_DIR,
    SOURCE_DATA_DIR,
    load_arff,
    write_csv,
)

LAB2_DIR = os.path.join(LAB1_DIR, "..", "lab2")
LAB2_RESULTS = os.path.join(LAB2_DIR, "results")
LAB2_FIGURES = os.path.join(LAB2_DIR, "figures")

#: The generating means stated on p.29 of the deck.
TRUE_MEANS = np.array([[3.0, 3.0], [7.0, 7.0]])

SEED = 42


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def load_gauss(name: str):
    """Load one gaussian file; returns (X, y_true_int) with labels 1..2 -> 0..1."""
    X, y_str, attrs, classes, _rel = load_arff(os.path.join(SOURCE_DATA_DIR, name))
    y = np.array([int(v) - 1 for v in y_str], dtype=int)
    return X, y, attrs, classes


def load_digits():
    from arff_utils import digit_path

    X, y_str, attrs, classes, _ = load_arff(digit_path(1))
    return X, np.array([int(v) for v in y_str]), attrs, classes


def fit_kmeans(X: np.ndarray, k: int, seed: int = SEED, n_init: int = 10):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return km


def centroid_report(X: np.ndarray, km, true_means: np.ndarray = TRUE_MEANS) -> list:
    """Distance from each learned centroid to the nearest generating mean."""
    rows = []
    for i, c in enumerate(km.cluster_centers_):
        # The gaussian data is labelled 1/2; centroid order is arbitrary, so
        # report the distance to the *closest* generating mean.
        d = np.linalg.norm(true_means - c, axis=1)
        j = int(np.argmin(d))
        rows.append({
            "centroid_index": i,
            "centroid_x": float(c[0]),
            "centroid_y": float(c[1]),
            "n_assigned": int((km.labels_ == i).sum()),
            "closest_gaussian_mean": f"({true_means[j][0]:.1f}, {true_means[j][1]:.1f})",
            "distance_to_closest_mean": float(d[j]),
            "distance_to_other_mean": float(d[1 - j]),
        })
    return rows


def classes_to_clusters(y_true: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """Assign each cluster to the majority class inside it.

    This is exactly WEKA's "classes to clusters evaluation": clusters are mapped
    to classes by majority vote, then a confusion matrix and accuracy are formed.
    """
    n_classes = int(y_true.max()) + 1
    table = np.zeros((n_classes, k), dtype=int)
    for c in range(n_classes):
        for j in range(k):
            table[c, j] = int(np.sum((y_true == c) & (labels == j)))

    # majority class per cluster
    cluster_to_class = np.argmax(table, axis=0)
    mapped = cluster_to_class[labels]
    accuracy = float(np.mean(mapped == y_true))

    # confusion matrix: rows = true class, cols = assigned class
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for c in range(n_classes):
        for j in range(k):
            conf[c, cluster_to_class[j]] += table[c, j]

    return {
        "class_by_cluster_table": table.tolist(),
        "cluster_to_class": [int(v) for v in cluster_to_class],
        "accuracy": accuracy,
        "n_errors": int(np.sum(mapped != y_true)),
        "confusion_matrix": conf.tolist(),
    }


def overfitting_estimate(y_true: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """The p.42 shortcut: estimate accuracy without re-labelling unlabelled clusters.

    WEKA assigns exactly one cluster per class (the one containing the most
    samples of that class).  The remaining clusters are 'unlabelled'.  The true
    number of correct assignments can be recovered by subtracting, from the
    error count, the largest values in the columns of the classes-to-clusters
    matrix that belong to unlabelled clusters.
    """
    n_classes = int(y_true.max()) + 1
    table = np.zeros((n_classes, k), dtype=int)
    for c in range(n_classes):
        for j in range(k):
            table[c, j] = int(np.sum((y_true == c) & (labels == j)))

    majorities = np.argmax(table, axis=0)          # class winning each cluster
    col_max = table.max(axis=0)                    # votes for that winner
    labelled = set()
    for c in range(n_classes):
        # the cluster with the most votes for class c is the one labelled c
        cand = [j for j in range(k) if majorities[j] == c]
        if cand:
            best = max(cand, key=lambda j: table[c, j])
            labelled.add(best)

    errors_reported = int(np.sum(y_true != majorities[labels]))
    recovered = int(sum(col_max[j] for j in range(k) if j not in labelled))
    return {
        "n_clusters": k,
        "n_labeled_clusters": len(labelled),
        "n_unlabeled_clusters": k - len(labelled),
        "weka_reported_errors": errors_reported,
        "recoverable_from_unlabeled_clusters": recovered,
        "naive_accuracy": float(1 - errors_reported / len(y_true)),
    }


# --------------------------------------------------------------------------
# exercises
# --------------------------------------------------------------------------


def ex_2_1_to_2_3(out: dict) -> None:
    print("\n" + "=" * 72)
    print("LAB 2, exercises 2.1-2.3: K-Means on gausstrain.arff, k=2")
    print("=" * 72)

    X, y, attrs, classes = load_gauss("gausstrain.arff")
    print(f"data: {X.shape}, class counts {np.bincount(y).tolist()} (0-indexed)")
    print(f"true generating means (p.29): {TRUE_MEANS.tolist()}")

    km = fit_kmeans(X, 2)
    print(f"\nK-Means with k=2, inertia = {km.inertia_:.4f}")

    rows = centroid_report(X, km)
    print("\ncentroid placement:")
    for r in rows:
        print(f"  centroid {r['centroid_index']}: ({r['centroid_x']:.4f}, {r['centroid_y']:.4f}) "
              f"n={r['n_assigned']:3d}  -> nearest true mean {r['closest_gaussian_mean']} "
              f"at distance {r['distance_to_closest_mean']:.4f}")

    # Empirical mean of each ground-truth class, for reference: the *best* any
    # clustering could do is place centroids at these positions.
    print("\nfor reference, the empirical class means are:")
    empirical = []
    for c in (0, 1):
        m = X[y == c].mean(axis=0)
        empirical.append([float(m[0]), float(m[1])])
        print(f"  class {c+1}: ({m[0]:.4f}, {m[1]:.4f})  n={int((y==c).sum())}")
    emp = np.array(empirical)
    for r in rows:
        c = np.array([r["centroid_x"], r["centroid_y"]])
        r["distance_to_empirical_class_mean"] = float(
            np.min(np.linalg.norm(emp - c, axis=1)))

    c2c = classes_to_clusters(y, km.labels_, 2)
    print("\n--- classes-to-clusters evaluation (exercise 2.3) ---")
    print(f"classes x clusters table (rows=true class 1,2; cols=clusters):")
    for i, row in enumerate(c2c["class_by_cluster_table"]):
        print(f"  class {i+1}: {row}")
    print(f"cluster -> class mapping: {c2c['cluster_to_class']}")
    print(f"accuracy = {c2c['accuracy']:.4f}  ({c2c['n_errors']} errors of {len(y)})")
    print(f"confusion matrix (rows=true, cols=assigned): {c2c['confusion_matrix']}")

    # Which points are misassigned, and are they the ones on the 'wrong' side?
    cluster_to_class = np.argmax(np.array(c2c["class_by_cluster_table"]), axis=0)
    wrong = km.labels_ != cluster_to_class[y]
    wrong_idx = np.flatnonzero(wrong)
    print(f"\nmisclassified points: {len(wrong_idx)}")
    if len(wrong_idx):
        # how far are they from their own class mean vs the other?
        for i in wrong_idx[:10]:
            d0 = np.linalg.norm(X[i] - emp[0])
            d1 = np.linalg.norm(X[i] - emp[1])
            print(f"  idx {i:3d} ({X[i,0]:6.3f},{X[i,1]:6.3f}) true class {y[i]+1}, "
                  f"d(mean1)={d0:.3f} d(mean2)={d1:.3f}")
        if len(wrong_idx) > 10:
            print(f"  ... and {len(wrong_idx)-10} more")
        centre = emp.mean(axis=0)
        near_boundary = np.sum(
            np.abs(np.linalg.norm(X[wrong_idx] - emp[0], axis=1)
                   - np.linalg.norm(X[wrong_idx] - emp[1], axis=1)) < 1.0)
        print(f"  of these, {near_boundary} are within 1.0 of the perpendicular "
              f"bisector between the two means")

    out["ex_2_1_2_3"] = {
        "inertia": float(km.inertia_),
        "centroids": rows,
        "empirical_class_means": empirical,
        "classes_to_clusters": c2c,
        "n_misclassified": int(len(wrong_idx)),
    }

    write_csv(os.path.join(LAB2_RESULTS, "kmeans_gausstrain_centroids.csv"), rows,
              ["centroid_index", "centroid_x", "centroid_y", "n_assigned",
               "closest_gaussian_mean", "distance_to_closest_mean",
               "distance_to_other_mean", "distance_to_empirical_class_mean"])
    print(f"\nwrote lab2/results/kmeans_gausstrain_centroids.csv")

    # -------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, mode in ((axes[0], "clusters"), (axes[1], "classes")):
        if mode == "clusters":
            for j in range(2):
                s = km.labels_ == j
                ax.scatter(X[s, 0], X[s, 1], s=22, alpha=0.75,
                           label=f"cluster {j+1} (n={int(s.sum())})")
            for i, c in enumerate(km.cluster_centers_):
                ax.plot(c[0], c[1], "k*", ms=20, mec="white", mew=1.2,
                        label="learned centroid" if i == 0 else None)
            ax.set_title("K-Means result: clusters and centroids")
        else:
            for c in (0, 1):
                s = y == c
                ax.scatter(X[s, 0], X[s, 1], s=22, alpha=0.75,
                           label=f"true class {c+1} (n={int(s.sum())})")
            wrong_pts = X[wrong]
            if len(wrong_pts):
                ax.scatter(wrong_pts[:, 0], wrong_pts[:, 1], s=90, facecolors="none",
                           edgecolors="red", linewidths=1.6,
                           label=f"misassigned ({len(wrong_pts)})")
            ax.set_title("Ground truth and the misassigned points")
        for i, m in enumerate(TRUE_MEANS):
            ax.plot(m[0], m[1], "kx", ms=14, mew=2.5,
                    label="true mean (3,3)/(7,7)" if i == 0 else None)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Lab 2 - K-Means on gausstrain.arff (k=2, 100 points)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB2_FIGURES, "kmeans_gausstrain.png"), dpi=150)
    plt.close(fig)
    print("wrote lab2/figures/kmeans_gausstrain.png")


def ex_2_4(out: dict) -> None:
    print("\n" + "=" * 72)
    print("LAB 2, exercise 2.4: gausstrainhv.arff (variance +50%)")
    print("=" * 72)

    results = {}
    for name in ("gausstrain.arff", "gausstrainhv.arff"):
        X, y, attrs, classes = load_gauss(name)
        emp = np.array([X[y == c].mean(axis=0) for c in (0, 1)])
        km = fit_kmeans(X, 2)
        rows = centroid_report(X, km)
        c2c = classes_to_clusters(y, km.labels_, 2)
        labels = "low variance" if "hv" not in name else "high variance (+50%)"
        print(f"\n--- {name}  ({labels}) ---")
        print(f"  empirical class variances: "
              f"{[round(float(X[y==c].var()),3) for c in (0,1)]}")
        for r in rows:
            print(f"  centroid {r['centroid_index']}: ({r['centroid_x']:.3f},{r['centroid_y']:.3f}) "
                  f"n={r['n_assigned']:3d}  dist to nearest true mean "
                  f"{r['distance_to_closest_mean']:.4f}")
        print(f"  classes-to-clusters accuracy = {c2c['accuracy']:.4f} "
              f"({c2c['n_errors']} errors)")

        # Distance of every point from the bisector between the true means: the
        # theoretical decision line of a 2-means solution with equal covariances.
        mid = TRUE_MEANS.mean(axis=0)
        direction = TRUE_MEANS[1] - TRUE_MEANS[0]
        direction = direction / np.linalg.norm(direction)
        signed = (X - mid) @ direction        # >0 towards (7,7), <0 towards (3,3)
        # a point is "geometrically ambiguous" if it lies on the wrong side of
        # the bisector relative to its own class mean
        proj_own = np.where(y == 0, -1.0, 1.0) * signed
        on_wrong_side = proj_own < 0
        print(f"  points on the far side of the bisector from their own mean: "
              f"{int(on_wrong_side.sum())}")
        # how many of THOSE does k-means still assign correctly?
        ctc = np.array(c2c["class_by_cluster_table"])
        cluster_to_class = np.argmax(ctc, axis=0)
        assigned_correct = cluster_to_class[km.labels_] == y
        print(f"  of those, k-means assigns {int((on_wrong_side & assigned_correct).sum())} "
              f"correctly anyway (they are closer to the other centroid)")
        print(f"  total misassigned = {c2c['n_errors']}")

        results[name] = {
            "centroids": rows,
            "classes_to_clusters": c2c,
            "n_points_on_wrong_side_of_bisector": int(on_wrong_side.sum()),
            "of_those_correctly_assigned": int((on_wrong_side & assigned_correct).sum()),
        }

        write_csv(os.path.join(LAB2_RESULTS, f"kmeans_{name.replace('.arff','')}_centroids.csv"),
                  rows, list(rows[0].keys()))

    out["ex_2_4"] = results

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, name in zip(axes, ("gausstrain.arff", "gausstrainhv.arff")):
        X, y, _, _ = load_gauss(name)
        km = fit_kmeans(X, 2)
        ctc = np.array(classes_to_clusters(y, km.labels_, 2)["class_by_cluster_table"])
        cluster_to_class = np.argmax(ctc, axis=0)
        wrong = cluster_to_class[km.labels_] != y
        for j in range(2):
            s = km.labels_ == j
            ax.scatter(X[s, 0], X[s, 1], s=22, alpha=0.7, label=f"cluster {j+1}")
        for c in km.cluster_centers_:
            ax.plot(c[0], c[1], "k*", ms=20, mec="white", mew=1.2)
        for m in TRUE_MEANS:
            ax.plot(m[0], m[1], "kx", ms=14, mew=2.5)
        if wrong.any():
            ax.scatter(X[wrong, 0], X[wrong, 1], s=95, facecolors="none",
                       edgecolors="red", linewidths=1.6, label=f"misassigned ({int(wrong.sum())})")
        # the bisector between the two generating means
        mid = TRUE_MEANS.mean(axis=0)
        d = TRUE_MEANS[1] - TRUE_MEANS[0]
        perp = np.array([-d[1], d[0]])
        perp = perp / np.linalg.norm(perp)
        t = np.linspace(-6, 6, 2)
        line = mid[None, :] + t[:, None] * perp[None, :]
        ax.plot(line[:, 0], line[:, 1], "k--", lw=1.2, label="bisector of true means")
        ax.set_title(f"{name}\naccuracy {classes_to_clusters(y, km.labels_, 2)['accuracy']:.3f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("y")
    fig.suptitle("Lab 2 - effect of increased variance on K-Means (k=2)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB2_FIGURES, "kmeans_variance_comparison.png"), dpi=150)
    plt.close(fig)
    print("\nwrote lab2/figures/kmeans_variance_comparison.png")


def ex_2_5(out: dict) -> None:
    print("\n" + "=" * 72)
    print("LAB 2, exercise 2.5: K-Means with k=10 on the digit images")
    print("=" * 72)

    X, y, attrs, classes = load_digits()
    print(f"data: {X.shape}, 10 digit classes")

    km = fit_kmeans(X, 10, n_init=10)
    c2c = classes_to_clusters(y, km.labels_, 10)
    print(f"inertia = {km.inertia_:.2f}")
    print(f"classes-to-clusters accuracy = {c2c['accuracy']:.4f} "
          f"({c2c['n_errors']} errors of {len(y)})")
    print("cluster -> digit mapping: "
          f"{c2c['cluster_to_class']}  (cluster i -> digit c2c['cluster_to_class'][i])")

    of = overfitting_estimate(y, km.labels_, 10)
    print(f"WEKA-style overfitting estimate: {of}")

    # Visualise the centroids as 13x8 greyscale images (p.39-40).
    fig, axes = plt.subplots(2, 5, figsize=(11, 6))
    for j, ax in enumerate(axes.ravel()):
        if j >= 10:
            ax.axis("off")
            continue
        # order clusters by their assigned digit for a readable layout
        c = km.cluster_centers_[j].reshape(13, 8)
        ax.imshow(c, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        n = int((km.labels_ == j).sum())
        ax.set_title(f"cluster {j}\nn={n}, grey={c.mean():.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Lab 2 - K-Means centroids on Bigtest1_104 (k=10), rendered as 13x8 images\n"
                 "grey level = probability that the pixel is ON within the cluster", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB2_FIGURES, "kmeans_digit_centroids.png"), dpi=150)
    plt.close(fig)
    print("wrote lab2/figures/kmeans_digit_centroids.png")

    # Do the centroids actually look like digits?  Compare each centroid to the
    # mean image of the digit class it was assigned to.
    ctc = np.array(c2c["class_by_cluster_table"])
    cluster_to_class = np.argmax(ctc, axis=0)
    sims = []
    for j in range(10):
        cls = cluster_to_class[j]
        mean_img = X[y == cls].mean(axis=0)
        # correlation between centroid and class-mean image
        r = np.corrcoef(km.cluster_centers_[j], mean_img)[0, 1]
        sims.append({"cluster": j, "assigned_digit": int(cls),
                     "n": int((km.labels_ == j).sum()),
                     "corr_with_class_mean": float(r)})
    print("\ncentroid similarity to the assigned digit's mean image:")
    for s in sims:
        print(f"  cluster {s['cluster']}: digit {s['assigned_digit']}  "
              f"corr = {s['corr_with_class_mean']:.4f}")
    mean_r = float(np.mean([s["corr_with_class_mean"] for s in sims]))
    print(f"  mean correlation = {mean_r:.4f}")

    write_csv(os.path.join(LAB2_RESULTS, "kmeans_digit_clusters.csv"), sims,
              ["cluster", "assigned_digit", "n", "corr_with_class_mean"])

    out["ex_2_5"] = {
        "inertia": float(km.inertia_),
        "classes_to_clusters": c2c,
        "overfitting_estimate": of,
        "centroid_similarity": sims,
        "mean_centroid_correlation": mean_r,
    }


def ex_2_6(out: dict) -> None:
    print("\n" + "=" * 72)
    print("LAB 2, exercise 2.6: how many clusters? (X-Means equivalent)")
    print("=" * 72)
    print("WEKA's XMeans was not available in this environment; the exercise is")
    print("reproduced with k-means over a range of k plus BIC and silhouette.")
    print("This is stated explicitly rather than presented as an XMeans result.\n")

    from sklearn.metrics import silhouette_score

    X_tr, y_tr, _, _ = load_gauss("gausstrain.arff")
    X_te, y_te, _, _ = load_gauss("gausstest.arff")
    Xd, yd, _, _ = load_digits()

    rows = []
    for name, X, y in (("gausstrain (2 gaussians)", X_tr, y_tr),
                       ("Bigtest1_104 (10 digits)", Xd, yd)):
        ks = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40] if "digit" in name else [2, 3, 4, 5, 6, 8]
        print(f"--- {name} ---")
        for k in ks:
            km = fit_kmeans(X, k, n_init=5)
            c2c = classes_to_clusters(y, km.labels_, k)
            sil = float(silhouette_score(X, km.labels_)) if k > 1 else float("nan")
            # BIC for k-means (Gaussian mixture with spherical equal-variance
            # components is the standard approximation)
            n, d = X.shape
            n_params = k * (d + 1)
            rss = float(np.sum((X - km.cluster_centers_[km.labels_]) ** 2))
            bic = n * np.log(rss / n) + n_params * np.log(n)
            rows.append({
                "dataset": name, "k": k, "inertia": float(km.inertia_),
                "bic": float(bic), "silhouette": sil,
                "classes_to_clusters_acc": c2c["accuracy"],
            })
            print(f"  k={k:3d}  inertia={km.inertia_:12.2f}  BIC={bic:10.1f}  "
                  f"silhouette={sil:+.4f}  classes-to-clusters acc={c2c['accuracy']:.4f}")

    write_csv(os.path.join(LAB2_RESULTS, "kmeans_k_selection.csv"), rows,
              ["dataset", "k", "inertia", "bic", "silhouette", "classes_to_clusters_acc"])

    # ---- does a finer partition generalise to the test set? (overfitting)
    print("\n--- generalisation of the partition size (train -> test) ---")
    n_tr = len(y_tr)
    ctc = None
    gen_rows = []
    for k in (2, 4, 6, 8, 12, 20):
        km = fit_kmeans(X_tr, k, n_init=5)
        # map each cluster to its majority class using the TRAINING labels only
        table = np.zeros((2, k), dtype=int)
        for c in (0, 1):
            for j in range(k):
                table[c, j] = int(np.sum((y_tr == c) & (km.labels_ == j)))
        cluster_to_class = np.argmax(table, axis=0)
        train_acc = float(np.mean(cluster_to_class[km.labels_] == y_tr))
        # assign test points to the nearest training centroid, using the same map
        d = np.linalg.norm(X_te[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        test_labels = np.argmin(d, axis=1)
        test_acc = float(np.mean(cluster_to_class[test_labels] == y_te))
        gen_rows.append({"k": k, "train_classes_to_clusters_acc": train_acc,
                         "test_acc_using_train_centroids": test_acc,
                         "gap_pp": 100 * (train_acc - test_acc)})
        print(f"  k={k:3d}  train acc={train_acc:.4f}  test acc={test_acc:.4f}  "
              f"gap={100*(train_acc-test_acc):+.2f} pp")
    write_csv(os.path.join(LAB2_RESULTS, "kmeans_generalisation.csv"), gen_rows,
              ["k", "train_classes_to_clusters_acc", "test_acc_using_train_centroids",
               "gap_pp"])

    out["ex_2_6"] = {
        "k_selection": rows,
        "generalisation": gen_rows,
        "note": ("XMeans unavailable in this environment; k-means over a range of k "
                 "with BIC and silhouette is used as a documented substitute."),
    }

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, ylabel in ((axes[0], "bic", "BIC (lower is better)"),
                            (axes[1], "silhouette", "silhouette (higher is better)")):
        for name in {r["dataset"] for r in rows}:
            sub = [r for r in rows if r["dataset"] == name and r[key] == r[key]]
            ax.plot([r["k"] for r in sub], [r[key] for r in sub], "o-", label=name)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Lab 2 - cluster-count selection criteria", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB2_FIGURES, "kmeans_k_selection.png"), dpi=150)
    plt.close(fig)
    print("wrote lab2/figures/kmeans_k_selection.png")


def main() -> None:
    os.makedirs(LAB2_RESULTS, exist_ok=True)
    os.makedirs(LAB2_FIGURES, exist_ok=True)

    out = {"lab": 2, "title": "Clustering"}
    ex_2_1_to_2_3(out)
    ex_2_4(out)
    ex_2_5(out)
    ex_2_6(out)

    with open(os.path.join(LAB2_RESULTS, "lab2_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote lab2/results/lab2_summary.json")


if __name__ == "__main__":
    main()
