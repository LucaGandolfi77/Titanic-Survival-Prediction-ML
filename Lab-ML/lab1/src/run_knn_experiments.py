"""Plan sections 3.2-3.4: kNN k-sweep, validation-based selection,
decision-boundary figures and ground-truth verification on the dense grid.

Outputs
-------
results/knn_k_sweep.csv          k vs train/test/LOO/CV accuracy
results/knn_cv_folds.json        per-fold accuracies for 5- and 10-fold CV
figures/knn_k_sweep.png          accuracy vs k (train, LOO, 5-CV, test, sklearn)
figures/knn_boundary.png         decision regions for k in {1,5,15,31}
figures/knn_errors_kstar.png     grid classification errors at k*
results/knn_summary.json         headline numbers used by the report
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
    FIGURES_DIR,
    RESULTS_DIR,
    load_circle,
)
from knn_scratch import (  # noqa: E402
    accuracy,
    cv_accuracy,
    knn_predict,
    loo_predictions,
)

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 41, 51, 61, 81, 99]


def true_labels_circle(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth for the original problem: c iff inside the unit circle.

    Returns (labels, on_boundary_mask).  The second value flags points that sit
    exactly on the circle or on the square edge, where the label is arbitrary.
    """
    x, y = X[:, 0], X[:, 1]
    r2 = x ** 2 + y ** 2
    inside = r2 <= 1.0
    labels = np.where(inside, 0, 1)  # c=0, q=1
    on_boundary = (np.abs(r2 - 1.0) < 1e-9) | (
        np.abs(np.maximum(np.abs(x), np.abs(y)) - 1.25) < 1e-9
    )
    return labels.astype(int), on_boundary


def true_labels_square_inner(X: np.ndarray, half_side: float = 0.88) -> np.ndarray:
    """Ground truth for the Exercise 1b relabelling: c iff inside the inner square."""
    inside = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) <= half_side + 1e-12
    return np.where(inside, 0, 1).astype(int)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_tr, y_tr, attrs = load_circle("train")
    X_te, y_te, _ = load_circle("test")
    X_all, y_all, _ = load_circle("all")

    # Majority-class (trivial) baselines: every accuracy figure is read against
    # these, since the circle problem is close to balanced but not exactly so.
    train_majority = int(np.bincount(y_tr).argmax())
    test_majority = int(np.bincount(y_te).argmax())
    train_majority_acc = float(np.mean(y_tr == train_majority))
    test_majority_acc = float(np.mean(y_te == test_majority))
    all_majority_acc = float(np.mean(y_all == int(np.bincount(y_all).argmax())))
    print(
        f"majority-class baselines: train={train_majority_acc:.4f} "
        f"(class {train_majority}), test={test_majority_acc:.4f} "
        f"(class {test_majority}), grid={all_majority_acc:.4f}"
    )

    from sklearn.neighbors import KNeighborsClassifier

    from knn_scratch import cv_accuracy_multi_k

    cv5_all = cv_accuracy_multi_k(X_tr, y_tr, K_VALUES, n_folds=5, seed=0)
    cv10_all = cv_accuracy_multi_k(X_tr, y_tr, K_VALUES, n_folds=10, seed=0)

    rows = []
    loo_preds = {}
    for k in K_VALUES:
        pred_tr = knn_predict(X_tr, y_tr, X_tr, k)
        pred_te = knn_predict(X_tr, y_tr, X_te, k)
        loo_pred = loo_predictions(X_tr, y_tr, k)
        loo_preds[k] = loo_pred
        cv5_mean, cv5_std, _ = cv5_all[k]
        cv10_mean, cv10_std, _ = cv10_all[k]
        sk = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)
        sk_acc = accuracy(y_te, sk)

        rows.append(
            {
                "k": k,
                "train_acc": accuracy(y_tr, pred_tr),
                "test_acc": accuracy(y_te, pred_te),
                "loo_acc": accuracy(y_tr, loo_pred),
                "cv5_acc": cv5_mean,
                "cv5_std": cv5_std,
                "cv10_acc": cv10_mean,
                "cv10_std": cv10_std,
                "sklearn_test_acc": sk_acc,
                "sklearn_agrees": int(np.sum(sk == pred_te)),
                "all_acc": accuracy(y_all, knn_predict(X_tr, y_tr, X_all, k)),
            }
        )
        print(
            f"k={k:>3}  train={rows[-1]['train_acc']:.3f}  loo={rows[-1]['loo_acc']:.3f} "
            f" cv5={cv5_mean:.3f}+-{cv5_std:.3f}  test={rows[-1]['test_acc']:.3f}"
            f"  all={rows[-1]['all_acc']:.3f}"
        )

    # ---------------------------------------------------------------- CSV
    csv_path = os.path.join(RESULTS_DIR, "knn_k_sweep.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {csv_path}")

    # ------------------------------------------------- hyper-parameter choice
    def best_k(key, rows=rows):
        best = max(rows, key=lambda r: (r[key], -r["k"]))
        return best["k"], best[key]

    k_test, acc_test = best_k("test_acc")
    k_loo, acc_loo = best_k("loo_acc")
    k_cv5, acc_cv5 = best_k("cv5_acc")
    k_cv10, acc_cv10 = best_k("cv10_acc")

    # k* chosen honestly: by LOO on the training set only.
    k_star = k_loo

    print(
        f"\nbest k by test set   : k={k_test} ({acc_test:.4f})   <- NOT used to choose k*\n"
        f"best k by LOO-CV     : k={k_loo} ({acc_loo:.4f})   <- k* (honest)\n"
        f"best k by 5-fold CV  : k={k_cv5} ({acc_cv5:.4f})\n"
        f"best k by 10-fold CV : k={k_cv10} ({acc_cv10:.4f})\n"
        f"chosen k* = {k_star}"
    )

    # ------------------------------------------------- CV fold details
    fold_details = {}
    for nf, cache in ((5, cv5_all), (10, cv10_all)):
        per_fold = cache[k_star][2]
        fold_details[f"cv{nf}_at_kstar"] = {
            "k": k_star,
            "folds": per_fold,
            "mean": float(np.mean([f["accuracy"] for f in per_fold])),
            "std": float(np.std([f["accuracy"] for f in per_fold])),
        }
    for nf, cache in ((5, cv5_all), (10, cv10_all)):
        fold_details[f"cv{nf}_sweep"] = [
            {"k": k, "accuracies": [f["accuracy"] for f in cache[k][2]]}
            for k in K_VALUES
        ]
    with open(os.path.join(RESULTS_DIR, "knn_cv_folds.json"), "w", encoding="utf-8") as fh:
        json.dump(fold_details, fh, indent=2)

    # Stability of argmax across folds: fraction of folds in which each k wins.
    k_stability = {}
    for nf in (5, 10):
        entry = fold_details[f"cv{nf}_sweep"]
        n_folds = len(entry[0]["accuracies"])
        wins = {k: 0 for k in K_VALUES}
        for fi in range(n_folds):
            accs = {e["k"]: e["accuracies"][fi] for e in entry}
            winner = max(accs, key=lambda kk: (accs[kk], -kk))
            wins[winner] += 1
        k_stability[f"cv{nf}_fold_winners"] = wins
        print(f"{nf}-fold CV per-fold winners: {wins}")

    # ------------------------------------------------------------- figure 1
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ks, [r["train_acc"] for r in rows], "o-", label="train accuracy", color="#1f77b4")
    ax.plot(ks, [r["loo_acc"] for r in rows], "s-", label="leave-one-out CV (train)", color="#2ca02c")
    ax.errorbar(
        ks, [r["cv5_acc"] for r in rows], yerr=[r["cv5_std"] for r in rows],
        fmt="^-", capsize=3, label="5-fold stratified CV (train)", color="#ff7f0e", alpha=0.85,
    )
    ax.plot(ks, [r["test_acc"] for r in rows], "d-", label="test accuracy (circletest)", color="#d62728")
    ax.plot(
        ks, [r["sklearn_test_acc"] for r in rows], "k--", lw=1, alpha=0.8,
        label="sklearn test accuracy (check)",
    )
    ax.axvline(k_star, color="grey", ls=":", lw=1.5)
    ax.annotate(
        f"$k^*={k_star}$ (by LOO-CV)\ntest={dict(zip([r['k'] for r in rows], [r['test_acc'] for r in rows]))[k_star]:.2f}",
        xy=(k_star, 0.5), xytext=(k_star + 6, 0.42),
        arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9, color="grey",
    )
    ax.axhline(test_majority_acc, color="black", ls="-.", lw=1, alpha=0.6,
               label=f"majority-class baseline, test ({test_majority_acc:.2f})")
    ax.axhline(1.0 / 2, color="grey", ls=":", lw=1, alpha=0.6,
               label="chance (2 classes)")
    ax.set_xlabel("k (number of neighbours, odd)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.03)
    ax.set_title("kNN on the circle problem: accuracy vs k")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "knn_k_sweep.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/knn_k_sweep.png")

    # ------------------------------------------------------------- figure 2
    boundary_ks = [1, 5, 15, 31]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, k in zip(axes, boundary_ks):
        pred = knn_predict(X_tr, y_tr, X_all, k)
        grid = pred.reshape(51, 51)
        ax.pcolormesh(
            np.linspace(-1.25, 1.25, 51), np.linspace(-1.25, 1.25, 51), grid,
            cmap="coolwarm", shading="auto", vmin=-0.3, vmax=1.3, alpha=0.85,
        )
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.8, label="true circle $r=1$")
        ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="k", lw=1.8,
                               ls="--", label="true square $|x|,|y|=1.25$"))
        # decision errors against the *file's own* labels
        wrong = pred != y_all
        ax.scatter(X_all[wrong, 0], X_all[wrong, 1], s=14, c="lime", edgecolors="k",
                   linewidths=0.4, zorder=5, label=f"errors vs grid labels ({wrong.sum()})")
        ax.scatter(X_tr[:, 0], X_tr[:, 1], s=12, c="k", marker=".", zorder=6,
                   label="training points" if k == boundary_ks[0] else None)
        ax.set_title(f"k = {k}   train-subset accuracy = {accuracy(y_all, pred):.3f}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("kNN decision regions on the dense grid (circleall.arff): red = q, blue = c",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "knn_boundary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/knn_boundary.png")

    # ------------------------------------------- ground-truth verification
    y_true, on_boundary = true_labels_circle(X_all)
    agreement = y_true == np.asarray(y_all, dtype=int)
    n_disagree = int((~agreement).sum())
    n_disagree_off_boundary = int((~agreement & ~on_boundary).sum())
    n_boundary = int(on_boundary.sum())
    print(
        f"\ncircleall ground truth vs file labels: {len(y_true) - n_disagree}/{len(y_true)} agree; "
        f"{n_disagree} disagreements ({n_disagree_off_boundary} off-boundary), "
        f"{n_boundary} points sit exactly on a boundary"
    )

    k_star_pred = knn_predict(X_tr, y_tr, X_all, k_star)
    acc_vs_file = accuracy(y_all, k_star_pred)
    acc_vs_true = accuracy(y_true, k_star_pred)
    mask = ~on_boundary
    acc_vs_true_offbound = accuracy(y_true[mask], k_star_pred[mask])
    errors_at_kstar = int((k_star_pred != y_all).sum())
    print(
        f"k*={k_star}: accuracy vs file labels={acc_vs_file:.4f}, vs ground truth={acc_vs_true:.4f} "
        f"({acc_vs_true_offbound:.4f} excluding on-boundary points), errors={errors_at_kstar}"
    )

    # per-k ground-truth accuracy table
    grid_rows = []
    for r in rows:
        pred = knn_predict(X_tr, y_tr, X_all, r["k"])
        grid_rows.append(
            {
                "k": r["k"],
                "grid_acc_vs_ground_truth": accuracy(y_true, pred),
                "grid_acc_vs_ground_truth_offboundary": accuracy(y_true[mask], pred[mask]),
                "grid_errors": int((pred != np.asarray(y_all, dtype=int)).sum()),
            }
        )
    with open(os.path.join(RESULTS_DIR, "knn_grid_accuracy.csv"), "w", encoding="utf-8") as fh:
        fh.write("k,grid_acc_vs_ground_truth,grid_acc_vs_ground_truth_offboundary,grid_errors\n")
        for g in grid_rows:
            fh.write(
                f"{g['k']},{g['grid_acc_vs_ground_truth']},"
                f"{g['grid_acc_vs_ground_truth_offboundary']},{g['grid_errors']}\n"
            )

    # ----------------------------------------------- figure 3: error map
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ok = k_star_pred == np.asarray(y_all, dtype=int)
    ax.scatter(X_all[ok, 0], X_all[ok, 1], s=10, c="lightgrey", label="correct")
    ax.scatter(X_all[~ok, 0], X_all[~ok, 1], s=26, c="red", edgecolors="k",
               linewidths=0.4, label=f"misclassified ({int((~ok).sum())})")
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.6)
    ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="k", lw=1.6, ls="--"))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"kNN errors on circleall.arff at $k^*={k_star}$ (grid accuracy "
                 f"{acc_vs_file:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "knn_errors_kstar.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/knn_errors_kstar.png")

    summary = {
        "k_values": K_VALUES,
        "k_star_by_loo": k_star,
        "loo_acc_at_kstar": acc_loo,
        "k_best_test": k_test,
        "test_acc_at_k_best_test": acc_test,
        "test_acc_at_kstar": dict(zip([r["k"] for r in rows], [r["test_acc"] for r in rows]))[k_star],
        "k_best_cv5": k_cv5,
        "cv5_acc_at_k_best_cv5": acc_cv5,
        "k_best_cv10": k_cv10,
        "cv10_acc_at_k_best_cv10": acc_cv10,
        "majority_baselines": {
            "train": train_majority_acc,
            "test": test_majority_acc,
            "grid": all_majority_acc,
        },
        "k_stability": k_stability,
        "circleall_ground_truth": {
            "n_points": int(len(y_true)),
            "n_boundary_points": n_boundary,
            "n_disagreements_with_file": n_disagree,
            "n_disagreements_off_boundary": n_disagree_off_boundary,
            "acc_at_kstar_vs_file": acc_vs_file,
            "acc_at_kstar_vs_truth": acc_vs_true,
            "acc_at_kstar_vs_truth_offboundary": acc_vs_true_offbound,
            "errors_at_kstar": errors_at_kstar,
        },
        "train_fit_perfect_at_k1": bool(rows[0]["train_acc"] == 1.0),
        "sklearn_agreement": all(r["sklearn_agrees"] == 100 for r in rows),
    }
    with open(os.path.join(RESULTS_DIR, "knn_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote results/knn_summary.json")


if __name__ == "__main__":
    main()
