"""main.py — Entry point: runs all clustering lab parts sequentially."""

from __future__ import annotations

import numpy as np

from arff_parser import load_arff
from kmeans import KMeans
from xmeans import XMeans
from evaluation import (
    classes_to_clusters,
    cluster_accuracy,
    confusion_matrix,
    print_confusion_matrix,
)
from visualization import (
    plot_gaussian_clusters,
    plot_digit_centroids,
    plot_overfitting_curve,
)


# =====================================================================
# PART A — 2-D Gaussian Clustering
# =====================================================================

def part_a_standard() -> None:
    """K-Means (k=2) on the standard-variance Gaussian dataset."""
    print("=" * 64)
    print("Part A — 2-D Gaussian Clustering (Standard Variance)")
    print("=" * 64)

    data = load_arff("data/gausstrain.arff")
    X, y = data.X, data.y

    km = KMeans(k=2).fit(X)

    print("\n  Centroids found:")
    for i, c in enumerate(km.centroids):
        print(f"    Cluster {i}: ({c[0]:.4f}, {c[1]:.4f})")
    print("  True Gaussian centres: (3, 3) and (7, 7)")

    # Classes-to-clusters evaluation
    mapping = classes_to_clusters(y, km.labels)
    acc, misclassified = cluster_accuracy(y, km.labels, mapping)
    print(f"\n  Cluster → Class mapping: {mapping}")
    print(f"  Classification accuracy: {acc:.2%}")
    print(f"  Misclassified samples  : {misclassified.sum()}")

    cm, cls = confusion_matrix(y, km.labels, mapping)
    print("\n  Confusion matrix:")
    print_confusion_matrix(cm, cls)

    plot_gaussian_clusters(
        X, km.labels, km.centroids,
        title="K-Means (k=2) on Gaussian Train (Standard)",
        save_name="gaussian_clusters.png",
        misclassified=misclassified,
    )


def part_a_high_variance() -> None:
    """K-Means (k=2) on the high-variance Gaussian dataset."""
    print("\n" + "=" * 64)
    print("Part A — 2-D Gaussian Clustering (High Variance, +50%)")
    print("=" * 64)

    data = load_arff("data/gausstrainhv.arff")
    X, y = data.X, data.y

    km = KMeans(k=2).fit(X)

    print("\n  Centroids found:")
    for i, c in enumerate(km.centroids):
        print(f"    Cluster {i}: ({c[0]:.4f}, {c[1]:.4f})")

    mapping = classes_to_clusters(y, km.labels)
    acc, misclassified = cluster_accuracy(y, km.labels, mapping)
    print(f"\n  Cluster → Class mapping: {mapping}")
    print(f"  Classification accuracy: {acc:.2%}")
    print(f"  Misclassified samples  : {misclassified.sum()}")

    cm, cls = confusion_matrix(y, km.labels, mapping)
    print("\n  Confusion matrix:")
    print_confusion_matrix(cm, cls)

    plot_gaussian_clusters(
        X, km.labels, km.centroids,
        title="K-Means (k=2) on Gaussian Train (High Variance)",
        save_name="gaussian_hv_clusters.png",
        misclassified=misclassified,
    )

    print(
        "\n  Discussion:"
        "\n  With higher variance the two Gaussian distributions overlap more."
        "\n  Points near the decision boundary (roughly the line x+y = 10)"
        "\n  can legitimately belong to either distribution.  K-Means draws"
        "\n  a linear Voronoi boundary between centroids; samples on the"
        "\n  'wrong' side are labelled as misclassified, but they are not"
        "\n  truly wrong — they simply lie in the overlap region where"
        "\n  both clusters have non-negligible probability."
    )


# =====================================================================
# PART B — Digit Image Clustering
# =====================================================================

def part_b_digits() -> None:
    """K-Means (k=10) on 13×8 binary digit images."""
    print("\n" + "=" * 64)
    print("Part B — Digit Image Clustering (K-Means, k=10)")
    print("=" * 64)

    data = load_arff("data/Bigtest1_104.arff")
    X, y = data.X, data.y

    km = KMeans(k=10).fit(X)

    mapping = classes_to_clusters(y, km.labels)
    acc, _ = cluster_accuracy(y, km.labels, mapping)

    print(f"\n  Cluster → Class mapping: {mapping}")
    print(f"  Training accuracy (classes-to-clusters): {acc:.2%}")

    plot_digit_centroids(km.centroids, mapping)

    print(
        "\n  Note on centroid visualisation:"
        "\n  Each centroid pixel value is the mean of the binary feature"
        "\n  across all samples assigned to that cluster.  This equals the"
        "\n  probability that the pixel is ON for a random pattern in the"
        "\n  cluster.  Because digits within a cluster share a common shape,"
        "\n  the centroid looks like a blurred (probabilistic) version of"
        "\n  the digit — bright where the stroke usually appears, dark"
        "\n  where the background is."
    )


# =====================================================================
# PART C — X-Means Clustering + Overfitting Analysis
# =====================================================================

def _run_kmeans_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: list[int],
) -> tuple[list[int], list[float], list[float]]:
    """Run KMeans for each k and return (k, train_acc, test_acc) lists."""
    k_list: list[int] = []
    train_acc_list: list[float] = []
    test_acc_list: list[float] = []

    for k in k_values:
        km = KMeans(k=k, random_state=42).fit(X_train)
        mapping = classes_to_clusters(y_train, km.labels)
        train_acc, _ = cluster_accuracy(y_train, km.labels, mapping)

        test_labels = km.predict(X_test)
        # Ensure every test cluster has a mapping entry
        for cid in set(int(c) for c in test_labels):
            if cid not in mapping:
                dists = np.linalg.norm(
                    km.centroids[cid] - km.centroids[list(mapping.keys())],
                    axis=1,
                )
                nearest = list(mapping.keys())[int(np.argmin(dists))]
                mapping[cid] = mapping[nearest]

        test_acc, _ = cluster_accuracy(y_test, test_labels, mapping)
        k_list.append(k)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    return k_list, train_acc_list, test_acc_list


def part_c_xmeans() -> None:
    """X-Means + KMeans sweep on digit images with overfitting analysis."""
    print("\n" + "=" * 64)
    print("Part C — X-Means / KMeans on Digits (Overfitting Analysis)")
    print("=" * 64)

    train = load_arff("data/Bigtest1_104.arff")
    test  = load_arff("data/Bigtest2_104.arff")
    X_train, y_train = train.X, train.y
    X_test, y_test   = test.X, test.y

    # --- C1: X-Means with varying max_clusters ---
    max_k_list = [10, 20, 30, 40]
    xm_k_list: list[int] = []
    xm_train: list[float] = []
    xm_test: list[float] = []

    print(f"\n  X-Means (BIC-based splitting)")
    print(f"  {'max_k':>6} {'k_found':>8} {'Train Acc':>10} {'Test Acc':>10}")
    print("  " + "-" * 40)

    for max_k in max_k_list:
        xm = XMeans(min_clusters=10, max_clusters=max_k, random_state=42)
        xm.fit(X_train)

        mapping = classes_to_clusters(y_train, xm.labels)
        tr_acc, _ = cluster_accuracy(y_train, xm.labels, mapping)

        test_labels = xm.predict(X_test)
        for cid in set(int(c) for c in test_labels):
            if cid not in mapping:
                dists = np.linalg.norm(
                    xm.centroids[cid] - xm.centroids[list(mapping.keys())],
                    axis=1,
                )
                nearest = list(mapping.keys())[int(np.argmin(dists))]
                mapping[cid] = mapping[nearest]

        te_acc, _ = cluster_accuracy(y_test, test_labels, mapping)
        xm_k_list.append(xm.k)
        xm_train.append(tr_acc)
        xm_test.append(te_acc)

        print(f"  {max_k:>6} {xm.k:>8} {tr_acc:>10.2%} {te_acc:>10.2%}")

    # Confusion matrix for the largest X-Means
    last_xm = XMeans(min_clusters=10, max_clusters=max_k_list[-1],
                      random_state=42).fit(X_train)
    mapping = classes_to_clusters(y_train, last_xm.labels)
    test_labels = last_xm.predict(X_test)
    for cid in set(int(c) for c in test_labels):
        if cid not in mapping:
            dists = np.linalg.norm(
                last_xm.centroids[cid]
                - last_xm.centroids[list(mapping.keys())],
                axis=1,
            )
            nearest = list(mapping.keys())[int(np.argmin(dists))]
            mapping[cid] = mapping[nearest]
    print(f"\n  Confusion matrix (X-Means, max_k={max_k_list[-1]}, "
          f"k={last_xm.k}) on TEST set:")
    cm, cls = confusion_matrix(y_test, test_labels, mapping)
    print_confusion_matrix(cm, cls)

    # --- C2: KMeans sweep for clear overfitting curve ---
    k_sweep = [10, 20, 30, 50, 75, 100, 150, 200]
    print(f"\n  K-Means sweep (direct k, for overfitting analysis)")
    print(f"  {'k':>6} {'Train Acc':>10} {'Test Acc':>10}")
    print("  " + "-" * 30)

    km_k, km_train, km_test = _run_kmeans_sweep(
        X_train, y_train, X_test, y_test, k_sweep,
    )
    for k, tr, te in zip(km_k, km_train, km_test):
        print(f"  {k:>6} {tr:>10.2%} {te:>10.2%}")

    # Plot overfitting curve using the KMeans sweep (clearer gap)
    plot_overfitting_curve(km_k, km_train, km_test)

    # Find best test k
    best_test = max(km_test)
    best_idx = km_test.index(best_test)

    print(
        f"\n  Discussion:"
        f"\n  Best test accuracy: {best_test:.2%} at k={km_k[best_idx]}"
        f"\n"
        f"\n  As k increases, training accuracy improves because more"
        f"\n  clusters can capture finer group structure.  However, test"
        f"\n  accuracy eventually plateaus or drops — classic overfitting."
        f"\n  Over-specialised clusters fit noise in the training set and"
        f"\n  fail to generalise.  X-Means mitigates this via BIC-based"
        f"\n  stopping, automatically choosing a moderate k."
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    part_a_standard()
    part_a_high_variance()
    part_b_digits()
    part_c_xmeans()
    print("\n" + "=" * 64)
    print("All parts complete.  See outputs/ for saved plots.")
    print("=" * 64)


if __name__ == "__main__":
    main()
