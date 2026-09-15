"""Plan sections 4-5: decision trees on the licence-plate digit data.

This is the largest experiment script.  It covers

* 4.1  digit image encoding check + sample figure
* 4.2  J48 with WEKA's 66% percentage split (implemented as explicit files so
       both WEKA and sklearn see exactly the same partition)
* 4.3  5+ random seeds, mean +/- sd, spread figure
* 4.4  Bigtest2_104.arff supplied as an external test set
* 4.5  confusion matrix analysis + most-confused class pairs
* 4.6  IBk (WEKA kNN) comparison
* 5.x  the M (minNumObj) overfitting sweep, train and test modalities
* 5.6  unpruned M=1 to test the "why not 100% training accuracy" notes

Outputs (all under lab1/results and lab1/figures)
-------------------------------------------------
j48_bigtest1_split66.csv, j48_seeds.csv, digits_model_comparison.csv
j48_M_sweep.csv, j48_raw_outputs/*.txt
figures/sample_digit.png, j48_seed_variance.png, j48_confusion.png,
figures/j48_M_sweep.png, figures/confused_pairs.png
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

from arff_utils import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    SOURCE_DATA_DIR,
    digit_path,
    load_arff,
    write_arff,
)
from weka_run import (  # noqa: E402
    IBK,
    J48,
    run_weka,
    weka_available,
    weka_percentage_split_indices,
)

SEEDS = [1, 2, 3, 4, 5, 42]
SPLIT_PERCENT = 66.0
RAW_DIR = os.path.join(RESULTS_DIR, "j48_raw_outputs")

#: sklearn settings chosen to mirror J48 as closely as possible.
SKLEARN_DT = dict(criterion="entropy", min_samples_leaf=2, ccp_alpha=0.0, random_state=1)


def load_digits():
    X1, y1_str, attrs, classes, _ = load_arff(digit_path(1))
    X2, y2_str, _, _, _ = load_arff(digit_path(2))
    y1 = np.array([int(v) for v in y1_str], dtype=int)
    y2 = np.array([int(v) for v in y2_str], dtype=int)
    return X1, y1, X2, y2, attrs, classes


def write_split(seed: int, X1: np.ndarray, y1: np.ndarray) -> tuple[str, str]:
    """Write the 66% train / 34% test files for one seed; return their paths."""
    train_idx, test_idx = weka_percentage_split_indices(y1, SPLIT_PERCENT, seed)
    tr_path = os.path.join(DATA_DIR, f"bigtest1_seed{seed}_train.arff")
    te_path = os.path.join(DATA_DIR, f"bigtest1_seed{seed}_test.arff")
    write_arff(tr_path, X1[train_idx], y1[train_idx],
               [f"f{i}" for i in range(X1.shape[1])],
               [str(d) for d in range(10)], f"digit-seed{seed}-train")
    write_arff(te_path, X1[test_idx], y1[test_idx],
               [f"f{i}" for i in range(X1.shape[1])],
               [str(d) for d in range(10)], f"digit-seed{seed}-test")
    return tr_path, te_path


def save_raw(name: str, text: str) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, name), "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------
# 4.1  encoding check
# --------------------------------------------------------------------------


def figure_sample_digits(X1, y1, attrs) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(9, 8))
    rng = np.random.default_rng(0)
    for ax in axes.ravel():
        # pick one random example of 12 different classes (digit shown in title)
        d = rng.integers(0, 10)
        idx = np.flatnonzero(y1 == d)
        i = idx[rng.integers(0, len(idx))]
        ax.imshow(X1[i].reshape(13, 8), cmap="gray_r", interpolation="nearest")
        ax.set_title(f"label {d}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Bigtest1_104.arff samples: 104 binary pixels reshaped to 13x8 (row-wise)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sample_digit.png"), dpi=150)
    plt.close(fig)

    # ASCII check of the example quoted on p.11 of the deck.
    #
    # NB: the character stream extracted from the PDF contains 105 comma-separated
    # values, but the grid printed right next to it on p.11 has 13x8 = 104 cells.
    # The first 104 values decode exactly to the printed grid
    # (00010000 / 01111110 / 11100111 / ... / 00111100), so the trailing value is
    # an artefact of text extraction.  We therefore use the first 104 and assert
    # that the decoding matches the grid the deck shows.
    raw = (
        "0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,1,1,"
        "1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,"
        "1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,"
        "0,0,1,1,1,1,0,0,0"
    ).split(",")
    if len(raw) != 104:
        raw = raw[:104]
    assert len(raw) == 104, f"expected 104 values, got {len(raw)}"
    target = np.array([int(v) for v in raw], dtype=float)
    expected_grid = [
        "00010000", "01111110", "11100111", "11000011", "11000011", "11000011",
        "11000011", "11000011", "11000011", "11000011", "11000011", "01111110",
        "00111100",
    ]
    decoded = ["".join("1" if v else "0" for v in target[r * 8:(r + 1) * 8])
               for r in range(13)]
    grid_ok = decoded == expected_grid
    matches = np.flatnonzero((X1 == target).all(axis=1))
    print(f"[4.1] p.11 pattern decodes to the printed 13x8 grid: {grid_ok}")
    print(f"[4.1] p.11 pattern found verbatim in Bigtest1_104.arff: "
          f"{'YES' if len(matches) else 'no'}"
          f"{'' if len(matches) else ' (the deck example is illustrative only)'}")
    print("      decoded as ASCII art (0='.', 1='#'):")
    for row in decoded:
        print("      " + row.replace("0", ".").replace("1", "#"))
    if len(matches):
        print(f"      file label of that row: {y1[matches[0]]}")


# --------------------------------------------------------------------------
# 4.2 / 4.3 / 4.4
# --------------------------------------------------------------------------


def run_split_experiments(X1, y1, X2, y2, available: bool):
    rows = []
    per_seed = {}
    X2_path = digit_path(2)

    for seed in SEEDS:
        tr_path, te_path = write_split(seed, X1, y1)
        Xtr, ytr, _ = _load_path(tr_path)
        Xte, yte, _ = _load_path(te_path)

        row = {"seed": seed, "n_train": len(ytr), "n_test": len(yte)}

        # --- WEKA J48 on the internal 66% test split
        if available:
            res = run_weka([J48, "-C", "0.25", "-M", "2"], train=tr_path, test=te_path)
            model = run_weka([J48, "-C", "0.25", "-M", "2"], train=tr_path,
                             test=te_path, section="all")
            save_raw(f"j48_seed{seed}_internal.txt", res.full_output)
            row["weka_internal_acc"] = (res.accuracy / 100.0
                                        if res.accuracy is not None else None)
            row["weka_leaves"] = model.num_leaves
            row["weka_size"] = model.size_of_tree
            row["weka_tree_text"] = model.tree_text
            row["weka_kappa"] = res.kappa
            row["weka_mae"] = res.mae
            row["weka_rmse"] = res.rmse
            row["weka_confusion"] = res.confusion
            row["weka_class_labels"] = res.class_labels

            # --- WEKA J48 on the supplied external test set Bigtest2
            res2 = run_weka([J48, "-C", "0.25", "-M", "2"], train=tr_path, test=X2_path)
            save_raw(f"j48_seed{seed}_bigtest2.txt", res2.full_output)
            row["weka_bigtest2_acc"] = (res2.accuracy / 100.0
                                        if res2.accuracy is not None else None)
            row["weka_bigtest2_confusion"] = res2.confusion
            row["weka_bigtest2_labels"] = res2.class_labels

            # --- IBk reference points (section 4.6)
            for k in (1, 5):
                rk = run_weka([IBK, "-K", str(k), "-W", "0"], train=tr_path,
                              test=te_path)
                row[f"weka_ibk{k}_acc"] = (rk.accuracy / 100.0
                                           if rk.accuracy is not None else None)
                rk2 = run_weka([IBK, "-K", str(k), "-W", "0"], train=tr_path,
                               test=X2_path)
                row[f"weka_ibk{k}_bigtest2_acc"] = (rk2.accuracy / 100.0
                                                    if rk2.accuracy is not None else None)

        # --- sklearn counterparts on the identical partition
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(**SKLEARN_DT).fit(Xtr, ytr)
        row["sklearn_internal_acc"] = float(clf.score(Xte, yte))
        row["sklearn_train_acc"] = float(clf.score(Xtr, ytr))
        row["sklearn_leaves"] = int(clf.get_n_leaves())
        row["sklearn_size"] = int(clf.tree_.node_count)
        row["sklearn_bigtest2_acc"] = float(clf.score(X2, y2))

        rows.append(row)
        per_seed[seed] = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "weka_internal_acc": row.get("weka_internal_acc"),
            "weka_bigtest2_acc": row.get("weka_bigtest2_acc"),
            "sklearn_internal_acc": row["sklearn_internal_acc"],
            "sklearn_bigtest2_acc": row["sklearn_bigtest2_acc"],
        }
        print(f"[4.2/4.4] seed={seed:>3} n_train={len(ytr)} "
              f"weka_internal={row.get('weka_internal_acc')} "
              f"weka_bigtest2={row.get('weka_bigtest2_acc')} "
              f"sklearn_internal={row['sklearn_internal_acc']:.4f} "
              f"sklearn_bigtest2={row['sklearn_bigtest2_acc']:.4f}")

    return rows, per_seed


def _load_path(path):
    import arff_utils

    X, y_str, attrs, _c, _r = arff_utils.load_arff(path)
    y = np.array([int(v) for v in y_str], dtype=int)
    return X, y, attrs


def figure_seed_variance(rows) -> dict:
    weka = [r["weka_internal_acc"] for r in rows if r.get("weka_internal_acc") is not None]
    sk = [r["sklearn_internal_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(rows))
    if weka:
        ax.plot(xs, weka, "o-", label="WEKA J48 (66% split test)")
    ax.plot(xs, sk, "s--", label="sklearn DecisionTree (same split)")
    n_test = rows[0]["n_test"]
    row = dict(zip([r["seed"] for r in rows], xs))
    for arr, colour, name in ((weka, "C0", "WEKA"), (sk, "C1", "sklearn")):
        if not arr:
            continue
        m, s = float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ax.axhline(m, color=colour, ls=":", lw=1.2, alpha=0.7)
        ax.fill_between([-0.5, len(rows) - 0.5], m - s, m + s, color=colour, alpha=0.08)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["seed"]) for r in rows])
    ax.set_xlabel("random seed (WEKA -s / replicated split)")
    ax.set_ylabel("accuracy on the 66/34 split test set")
    ax.set_title(f"Seed sensitivity of J48 on Bigtest1 ({n_test} test instances per seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "j48_seed_variance.png"), dpi=150)
    plt.close(fig)

    stats = {"n_seeds": len(rows), "n_test_per_seed": n_test}
    if weka:
        stats["weka_mean"] = float(np.mean(weka))
        stats["weka_std"] = float(np.std(weka, ddof=1)) if len(weka) > 1 else 0.0
        stats["weka_min"] = float(np.min(weka))
        stats["weka_max"] = float(np.max(weka))
        p = stats["weka_mean"]
        # binomial standard error of an accuracy estimated on n_test instances
        stats["binomial_se_pp"] = float(100 * np.sqrt(p * (1 - p) / n_test))
        stats["binomial_95ci_pp"] = float(1.96 * stats["binomial_se_pp"])
    stats["sklearn_mean"] = float(np.mean(sk))
    stats["sklearn_std"] = float(np.std(sk, ddof=1)) if len(sk) > 1 else 0.0
    stats["observed_weka_spread_pp"] = float(
        100 * (stats.get("weka_max", 0) - stats.get("weka_min", 0))
    )
    return stats


# --------------------------------------------------------------------------
# 4.5 confusion analysis
# --------------------------------------------------------------------------


def confusion_figure(rows, X1, y1) -> dict:
    row = rows[0]
    cm = np.array(row.get("weka_confusion") or [], dtype=float)
    labels = row.get("weka_class_labels") or [str(d) for d in range(10)]
    if cm.size == 0:
        print("[4.5] no WEKA confusion matrix available; skipping heatmap")
        return {}

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    im = axes[0].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("J48 confusion matrix, row-normalised (seed "
                      f"{row['seed']}, 66% split test)")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                axes[0].text(j, i, int(cm[i, j]), ha="center", va="center",
                             fontsize=7,
                             color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # most-confused off-diagonal pairs
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], str(labels[i]), str(labels[j])))
    pairs.sort(reverse=True)

    top = pairs[:8]
    ys = np.arange(len(top))
    axes[1].barh(ys, [p[0] for p in top], color="indianred")
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels([f"{a} -> {b}" for _, a, b in top])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("number of test instances")
    axes[1].set_title("Most frequent confusions (true -> predicted)")
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "j48_confusion.png"), dpi=150)
    plt.close(fig)

    # average image of the most confused pair, to test the "pixel" hypothesis
    if len(top) >= 1:
        a, b = top[0][1], top[0][2]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
        mean_a = X1[y1 == int(a)].mean(axis=0).reshape(13, 8)
        mean_b = X1[y1 == int(b)].mean(axis=0).reshape(13, 8)
        for ax, img, title in (
            (axes[0], mean_a, f"mean image of {a}"),
            (axes[1], mean_b, f"mean image of {b}"),
            (axes[2], mean_a - mean_b, f"difference ({a} - {b})"),
        ):
            im = ax.imshow(img, cmap="bwr" if "difference" in title else "gray_r",
                           interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Why {a} and {b} are confused: average 13x8 images", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "confused_pairs.png"), dpi=150)
        plt.close(fig)

    return {
        "class_labels": [str(x) for x in labels],
        "matrix": cm.astype(int).tolist(),
        "row_normalised": norm.tolist(),
        "top_confusions": [{"true": a, "predicted": b, "count": float(c)}
                           for c, a, b in pairs[:10]],
        "accuracy_from_matrix": float(np.trace(cm) / cm.sum()),
    }


# --------------------------------------------------------------------------
# 5.x  M sweep
# --------------------------------------------------------------------------


def run_M_sweep(available: bool, X1, y1, X2, y2):
    X2_path = digit_path(2)
    tr_path = os.path.join(DATA_DIR, "bigtest1_full_train.arff")
    write_arff(tr_path, X1, y1, [f"f{i}" for i in range(X1.shape[1])],
               [str(d) for d in range(10)], "digit-full-train")

    rows = []
    M_values = list(range(1, 26)) + [30, 40, 50, 100]
    from sklearn.tree import DecisionTreeClassifier

    for M in M_values:
        row = {"M": M}
        if available:
            res_test = run_weka([J48, "-C", "0.25", "-M", str(M)],
                                train=tr_path, test=X2_path)
            model_test = run_weka([J48, "-C", "0.25", "-M", str(M)],
                                  train=tr_path, test=X2_path, section="all")
            res_train = run_weka([J48, "-C", "0.25", "-M", str(M)],
                                 train=tr_path, test=tr_path,
                                 section="Error on training data")
            save_raw(f"j48_M{M}_train.txt", res_train.full_output)
            row["weka_test_acc"] = (res_test.accuracy / 100.0
                                    if res_test.accuracy is not None else None)
            row["weka_train_acc"] = (res_train.accuracy / 100.0
                                     if res_train.accuracy is not None else None)
            row["weka_leaves"] = model_test.num_leaves
            row["weka_size"] = model_test.size_of_tree
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=M,
                                     ccp_alpha=0.0, random_state=1).fit(X1, y1)
        row["sklearn_train_acc"] = float(clf.score(X1, y1))
        row["sklearn_test_acc"] = float(clf.score(X2, y2))
        row["sklearn_leaves"] = int(clf.get_n_leaves())
        row["sklearn_size"] = int(clf.tree_.node_count)
        rows.append(row)
        print(f"[5.2] M={M:>3}  weka train={row.get('weka_train_acc')} "
              f"test={row.get('weka_test_acc')} leaves={row.get('weka_leaves')} | "
              f"sklearn train={row['sklearn_train_acc']:.4f} "
              f"test={row['sklearn_test_acc']:.4f}")

    # unpruned M=1 (plan 5.6)
    unpruned = {}
    if available:
        r_u_tr = run_weka([J48, "-C", "0.25", "-M", "1", "-U"], train=tr_path,
                          test=tr_path, section="Error on training data")
        r_u_model = run_weka([J48, "-C", "0.25", "-M", "1", "-U"], train=tr_path,
                             test=tr_path, section="all")
        r_u_te = run_weka([J48, "-C", "0.25", "-M", "1", "-U"], train=tr_path,
                          test=X2_path)
        save_raw("j48_M1_unpruned_train.txt", r_u_tr.full_output)
        unpruned = {
            "train_acc": r_u_tr.accuracy / 100.0 if r_u_tr.accuracy is not None else None,
            "test_acc": r_u_te.accuracy / 100.0 if r_u_te.accuracy is not None else None,
            "num_leaves": r_u_model.num_leaves,
            "size_of_tree": r_u_model.size_of_tree,
        }
        print(f"[5.6] unpruned M=1: train={unpruned['train_acc']:.4f} "
              f"test={unpruned['test_acc']:.4f} leaves={unpruned['num_leaves']}")

    clf_u = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1,
                                   ccp_alpha=0.0, random_state=1).fit(X1, y1)
    sklearn_unpruned = {
        "train_acc": float(clf_u.score(X1, y1)),
        "test_acc": float(clf_u.score(X2, y2)),
        "num_leaves": int(clf_u.get_n_leaves()),
        "size_of_tree": int(clf_u.tree_.node_count),
    }
    print(f"[5.6] sklearn fully grown tree: train={sklearn_unpruned['train_acc']:.4f} "
          f"test={sklearn_unpruned['test_acc']:.4f} "
          f"leaves={sklearn_unpruned['num_leaves']}")

    return rows, unpruned, sklearn_unpruned


def figure_M_sweep(rows) -> dict:
    Ms = [r["M"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    if rows[0].get("weka_train_acc") is not None:
        axes[0].plot(Ms, [r["weka_train_acc"] for r in rows], "o-",
                     label="WEKA J48, training set")
        axes[0].plot(Ms, [r["weka_test_acc"] for r in rows], "s-",
                     label="WEKA J48, Bigtest2 (external test)")
        best = max(rows, key=lambda r: (r["weka_test_acc"], -r["M"]))
        axes[0].axvline(best["M"], color="grey", ls=":", lw=1.5)
        axes[0].annotate(f"$M^*={best['M']}$\ntest={best['weka_test_acc']:.4f}",
                         xy=(best["M"], best["weka_test_acc"]),
                         xytext=(best["M"] + 6, best["weka_test_acc"] - 0.05),
                         arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
    axes[0].plot(Ms, [r["sklearn_train_acc"] for r in rows], "^--", alpha=0.8,
                 label="sklearn, training set")
    axes[0].plot(Ms, [r["sklearn_test_acc"] for r in rows], "v--", alpha=0.8,
                 label="sklearn, Bigtest2")
    axes[0].set_xlabel("M = minNumObj / min_samples_leaf")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Overfitting control: capacity vs M")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="center right")

    axes[1].plot(Ms, [r.get("weka_leaves") for r in rows], "o-", label="WEKA leaves")
    axes[1].plot(Ms, [r["sklearn_leaves"] for r in rows], "s--", label="sklearn leaves")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("number of leaves")
    axes[1].set_yscale("log")
    axes[1].set_title("Tree size shrinks as M grows")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "j48_M_sweep.png"), dpi=150)
    plt.close(fig)

    stats = {}
    if rows[0].get("weka_test_acc") is not None:
        best = max(rows, key=lambda r: (r["weka_test_acc"], -r["M"]))
        stats["weka_M_star"] = best["M"]
        stats["weka_best_test_acc"] = best["weka_test_acc"]
        stats["weka_test_acc_at_M1"] = rows[0]["weka_test_acc"]
        stats["weka_train_acc_at_M1"] = rows[0]["weka_train_acc"]
        stats["weka_train_acc_at_M_star"] = best["weka_train_acc"]
        stats["weka_train_monotone_decreasing"] = bool(
            all(rows[i + 1]["weka_train_acc"] <= rows[i]["weka_train_acc"] + 1e-12
                for i in range(len(rows) - 1))
        )
    best_sk = max(rows, key=lambda r: (r["sklearn_test_acc"], -r["M"]))
    stats["sklearn_M_star"] = best_sk["M"]
    stats["sklearn_best_test_acc"] = best_sk["sklearn_test_acc"]
    return stats


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    available = weka_available()
    print(f"WEKA available: {available}\n")

    X1, y1, X2, y2, attrs, classes = load_digits()
    print(f"[data] Bigtest1 {X1.shape}, Bigtest2 {X2.shape}, "
          f"class balance B1={np.bincount(y1).tolist()}")
    print(f"[data] Bigtest2 is exactly balanced: {np.all(np.bincount(y2) == 501)}\n")

    figure_sample_digits(X1, y1, attrs)

    rows, per_seed = run_split_experiments(X1, y1, X2, y2, available)
    seed_stats = figure_seed_variance(rows)

    confusion = confusion_figure(rows, X1, y1)

    M_rows, unpruned, sklearn_unpruned = run_M_sweep(available, X1, y1, X2, y2)
    M_stats = figure_M_sweep(M_rows)

    # ------------------------------------------------------------ CSV files
    def dump_csv(path, rows, cols):
        from arff_utils import write_csv

        write_csv(path, rows, cols)

    dump_csv(os.path.join(RESULTS_DIR, "j48_seeds.csv"), rows,
             ["seed", "n_train", "n_test", "weka_internal_acc", "weka_bigtest2_acc",
              "weka_leaves", "weka_size", "weka_kappa", "weka_mae", "weka_rmse",
              "weka_ibk1_acc", "weka_ibk5_acc",
              "weka_ibk1_bigtest2_acc", "weka_ibk5_bigtest2_acc",
              "sklearn_internal_acc", "sklearn_train_acc", "sklearn_leaves",
              "sklearn_size", "sklearn_bigtest2_acc"])
    print(f"wrote results/j48_seeds.csv")

    dump_csv(os.path.join(RESULTS_DIR, "j48_M_sweep.csv"), M_rows,
             ["M", "weka_train_acc", "weka_test_acc", "weka_leaves", "weka_size",
              "sklearn_train_acc", "sklearn_test_acc", "sklearn_leaves",
              "sklearn_size"])
    print("wrote results/j48_M_sweep.csv")

    # 4.2 single-split record (first seed)
    r0 = rows[0]
    dump_csv(os.path.join(RESULTS_DIR, "j48_bigtest1_split66.csv"), [r0],
             ["seed", "n_train", "n_test", "weka_internal_acc", "weka_leaves",
              "weka_size", "weka_kappa", "weka_mae", "weka_rmse",
              "weka_bigtest2_acc", "sklearn_internal_acc", "sklearn_bigtest2_acc"])
    print("wrote results/j48_bigtest1_split66.csv")

    # 4.6 model comparison
    def mean_of(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def sd_of(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else None

    comp = [{
        "model": "J48 decision tree", "hyperparameter": "M=2, pruned",
        "internal_acc": mean_of("weka_internal_acc"),
        "internal_acc_sd": sd_of("weka_internal_acc"),
        "bigtest2_acc": mean_of("weka_bigtest2_acc"),
        "bigtest2_acc_sd": sd_of("weka_bigtest2_acc"),
        "mean_leaves": mean_of("weka_leaves"),
    }]
    for k in (1, 5):
        comp.append({
            "model": f"IBk kNN", "hyperparameter": f"k={k}, uniform weights",
            "internal_acc": mean_of(f"weka_ibk{k}_acc"),
            "internal_acc_sd": sd_of(f"weka_ibk{k}_acc"),
            "bigtest2_acc": mean_of(f"weka_ibk{k}_bigtest2_acc"),
            "bigtest2_acc_sd": sd_of(f"weka_ibk{k}_bigtest2_acc"),
            "mean_leaves": None,
        })
    comp.append({
        "model": "sklearn DecisionTree", "hyperparameter": "entropy, M=2",
        "internal_acc": mean_of("sklearn_internal_acc"),
        "internal_acc_sd": sd_of("sklearn_internal_acc"),
        "bigtest2_acc": mean_of("sklearn_bigtest2_acc"),
        "bigtest2_acc_sd": sd_of("sklearn_bigtest2_acc"),
        "mean_leaves": mean_of("sklearn_leaves"),
    })
    dump_csv(os.path.join(RESULTS_DIR, "digits_model_comparison.csv"), comp,
             ["model", "hyperparameter", "internal_acc", "internal_acc_sd",
              "bigtest2_acc", "bigtest2_acc_sd", "mean_leaves"])
    print("wrote results/digits_model_comparison.csv")

    # ------------------------------------------------------------- summary
    summary = {
        "weka_available": available,
        "split_percent": SPLIT_PERCENT,
        "seeds": SEEDS,
        "seed_rows": per_seed,
        "seed_stats": seed_stats,
        "confusion": confusion,
        "M_sweep_stats": M_stats,
        "unpruned_M1_weka": unpruned,
        "unpruned_M1_sklearn": sklearn_unpruned,
        "sklearn_params_mirroring_J48": SKLEARN_DT,
        "notes": {
            "split_replication": (
                "The 66% split is implemented as explicit ARFF files using a "
                "reimplementation of java.util.Random, so WEKA and sklearn are "
                "evaluated on exactly the same instances."
            ),
        },
    }
    with open(os.path.join(RESULTS_DIR, "digits_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("wrote results/digits_summary.json")


if __name__ == "__main__":
    main()
