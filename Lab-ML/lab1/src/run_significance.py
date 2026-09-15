"""Significance testing and timing: closes the two gaps flagged in REPORT.md 7.5.

What this adds
--------------
* **Paired significance tests.** McNemar's test on identical test instances for
  every model pair, using *WEKA's own* predictions (via ``-p 0``) so the pairing
  is exact rather than approximated with a sklearn stand-in. Holm-Bonferroni
  correction is applied across the family of comparisons.
* **Wilson confidence intervals** on every headline accuracy, so the reader can
  see how much of each gap is sampling noise. On the 100-point circle test set
  the interval is ~8 pp wide - which is the whole point.
* **Inference-cost measurements** for kNN vs decision tree, replacing the empty
  ``train_time_s`` column and the analytic-only claim in section 5.3.

Outputs
-------
results/significance_digits_internal.csv   McNemar pairs, 66% split test set
results/significance_digits_bigtest2.csv   McNemar pairs, Bigtest2
results/significance_circle.csv            McNemar pairs, circle representations
results/accuracy_confidence_intervals.csv  accuracy + Wilson CI for every model
results/inference_benchmark.csv            fit / predict timings
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arff_utils import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    digit_path,
    load_arff,
    write_csv,
)
from stats_utils import (  # noqa: E402
    accuracy_with_ci,
    mcnemar_matrix,
)
from weka_run import (  # noqa: E402
    IBK,
    J48,
    run_weka_predictions,
    weka_available,
)

SPLIT_SEED = 1
BENCH_REPEATS = 5


def load_y(path: str) -> np.ndarray:
    X, y_str, _a, _c, _r = load_arff(path)
    # digit labels are plain integers; circle labels are c/q
    if y_str[0] in ("c", "q"):
        return np.array([0 if v == "c" else 1 for v in y_str], dtype=int)
    return np.array([int(v) for v in y_str], dtype=int)


def main() -> None:
    available = weka_available()
    print(f"WEKA available: {available}")
    if not available:
        print("!! WEKA missing: prediction capture requires WEKA. Aborting.")
        return

    tr = os.path.join(DATA_DIR, f"bigtest1_seed{SPLIT_SEED}_train.arff")
    te = os.path.join(DATA_DIR, f"bigtest1_seed{SPLIT_SEED}_test.arff")
    b2 = digit_path(2)
    y_te = load_y(te)
    y_b2 = load_y(b2)

    ci_rows = []
    significance = {}

    for label, test_path, y_test in (("digits_internal", te, y_te),
                                     ("digits_bigtest2", b2, y_b2)):
        preds = {}
        print(f"\n=== capturing WEKA predictions: {label} ===")
        specs = [
            ("J48 M=2 (pruned)", [J48, "-C", "0.25", "-M", "2"]),
            ("IBk k=1", [IBK, "-K", "1", "-W", "0"]),
            ("IBk k=5", [IBK, "-K", "5", "-W", "0"]),
            ("IBk k=11", [IBK, "-K", "11", "-W", "0"]),
        ]
        for name, args in specs:
            t0 = time.time()
            pred, actual = run_weka_predictions(args, tr, test_path)
            dt = time.time() - t0
            assert len(pred) == len(y_test), f"{name}: {len(pred)} vs {len(y_test)}"
            assert np.array_equal(actual, y_test), f"{name}: actual classes differ from file"
            preds[name] = pred
            stats = accuracy_with_ci(y_test, pred)
            ci_rows.append({
                "dataset": label, "model": name, "tool": "WEKA",
                **stats, "wall_clock_s_incl_jvm": round(dt, 2),
            })
            print(f"  {name:20s} acc={stats['accuracy']:.5f} "
                  f"CI95=[{stats['ci_low']:.5f},{stats['ci_high']:.5f}] "
                  f"({stats['ci_width_pp']:.2f}pp wide)  {dt:.1f}s incl. JVM start")

        # sklearn counterpart on the identical split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        X_tr, y_tr, _ = _load_xy(tr)
        X_te, _, _ = _load_xy(test_path)
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2,
                                     random_state=1).fit(X_tr, y_tr)
        preds["sklearn DT (entropy, M=2)"] = clf.predict(X_te)
        for k in (1, 5):
            kn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
            preds[f"sklearn kNN k={k}"] = kn.predict(X_te)

        for name, pred in preds.items():
            if name.startswith("sklearn"):
                stats = accuracy_with_ci(y_test, pred)
                ci_rows.append({"dataset": label, "model": name, "tool": "sklearn",
                                **stats, "wall_clock_s_incl_jvm": None})

        rows = mcnemar_matrix(y_test, preds, adjust=True)
        significance[label] = rows
        print(f"  --- McNemar pairwise ({len(rows)} comparisons, Holm-corrected) ---")
        for r in rows:
            mark = "*" if r.get("significant_05_holm") else " "
            print(f"   {mark} {r['model_a']:26s} vs {r['model_b']:26s} "
                  f"delta={r['delta_pp']:+6.2f}pp b={r['discordant_a_correct']:4d} "
                  f"c={r['discordant_b_correct']:4d} p={r['p_value']:.3g} "
                  f"p_holm={r.get('p_adjusted_holm', float('nan')):.3g}")

        write_csv(os.path.join(RESULTS_DIR, f"significance_{label}.csv"), rows,
                  ["model_a", "model_b", "acc_a", "acc_b", "delta_pp", "n_test",
                   "discordant_a_correct", "discordant_b_correct", "statistic",
                   "method", "p_value", "p_adjusted_holm", "significant_05_holm"])
        print(f"  wrote results/significance_{label}.csv")

    # ------------------------------------------------- circle representations
    print("\n=== circle representations: Wilson CIs and exact McNemar ===")
    circle_preds = {}
    circle_ci = []
    from sklearn.tree import DecisionTreeClassifier

    for variant in ("xy", "sq", "zt", "u"):
        tr_p = os.path.join(DATA_DIR, f"circletrain_{variant}.arff")
        te_p = os.path.join(DATA_DIR, f"circletest_{variant}.arff")
        y_c = load_y(te_p)
        # WEKA predictions for the u variant too (single attribute)
        pred, actual = run_weka_predictions([J48, "-C", "0.25", "-M", "2"], tr_p, te_p)
        assert np.array_equal(actual, y_c)
        circle_preds[f"J48 {variant}"] = pred
        stats = accuracy_with_ci(y_c, pred)
        circle_ci.append({"dataset": "circle", "model": f"J48 {variant}",
                          "tool": "WEKA", **stats, "wall_clock_s_incl_jvm": None})
        print(f"  J48 {variant:3s} acc={stats['accuracy']:.4f} "
              f"CI95=[{stats['ci_low']:.4f},{stats['ci_high']:.4f}] "
              f"({stats['ci_width_pp']:.1f}pp wide)")
        ci_rows.extend(circle_ci[-1:])

    circle_rows = mcnemar_matrix(y_c, circle_preds, adjust=True)
    for r in circle_rows:
        mark = "*" if r.get("significant_05_holm") else " "
        print(f"   {mark} {r['model_a']:8s} vs {r['model_b']:8s} "
              f"delta={r['delta_pp']:+6.2f}pp b={r['discordant_a_correct']:3d} "
              f"c={r['discordant_b_correct']:3d} p={r['p_value']:.3g} "
              f"p_holm={r.get('p_adjusted_holm', float('nan')):.3g}")
    write_csv(os.path.join(RESULTS_DIR, "significance_circle.csv"), circle_rows,
              ["model_a", "model_b", "acc_a", "acc_b", "delta_pp", "n_test",
               "discordant_a_correct", "discordant_b_correct", "statistic",
               "method", "p_value", "p_adjusted_holm", "significant_05_holm"])
    print("  wrote results/significance_circle.csv")

    write_csv(os.path.join(RESULTS_DIR, "accuracy_confidence_intervals.csv"), ci_rows,
              ["dataset", "model", "tool", "n", "n_correct", "accuracy",
               "ci_low", "ci_high", "ci_width_pp", "wall_clock_s_incl_jvm"])
    print("\nwrote results/accuracy_confidence_intervals.csv")

    # ------------------------------------------------------------- benchmark
    print("\n=== inference cost benchmark ===")
    bench_rows = benchmark(tr, b2)
    write_csv(os.path.join(RESULTS_DIR, "inference_benchmark.csv"), bench_rows,
              ["model", "role", "n_train", "n_test", "fit_s", "predict_s",
               "predict_us_per_instance", "notes"])
    for r in bench_rows:
        print(f"  {r['model']:28s} {r['role']:9s} fit={r['fit_s']:.4f}s "
              f"predict={r['predict_s']:.4f}s "
              f"({r['predict_us_per_instance']:.2f} us/instance)")
    print("wrote results/inference_benchmark.csv")


def _load_xy(path: str):
    X, y_str, attrs, _c, _r = load_arff(path)
    if y_str[0] in ("c", "q"):
        y = np.array([0 if v == "c" else 1 for v in y_str], dtype=int)
    else:
        y = np.array([int(v) for v in y_str], dtype=int)
    return X, y, attrs


def benchmark(train_path: str, test_path: str) -> list:
    """Time fitting and predicting for the tree and for kNN.

    The purpose is to substantiate the claim in REPORT.md section 5.3 that a tree
    is much cheaper to *apply* than 1-NN.  Timings are median-of-``BENCH_REPEATS``
    and include a warm-up call so that lazy import/allocation costs do not
    dominate the small numbers.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    X_tr, y_tr, _ = _load_xy(train_path)
    X_te, y_te, _ = _load_xy(test_path)

    models = {
        "DecisionTree (entropy, M=2)": DecisionTreeClassifier(
            criterion="entropy", min_samples_leaf=2, random_state=1),
        "kNN k=1": KNeighborsClassifier(n_neighbors=1),
        "kNN k=5": KNeighborsClassifier(n_neighbors=5),
    }

    rows = []
    for name, clf in models.items():
        # warm-up (also validates the model runs)
        clf.fit(X_tr, y_tr).predict(X_te[:16])
        fit_times, pred_times = [], []
        for _ in range(BENCH_REPEATS):
            t0 = time.perf_counter()
            model = clf.__class__(**clf.get_params()).fit(X_tr, y_tr)
            t1 = time.perf_counter()
            model.predict(X_te)
            t2 = time.perf_counter()
            fit_times.append(t1 - t0)
            pred_times.append(t2 - t1)
        fit_s = float(np.median(fit_times))
        pred_s = float(np.median(pred_times))
        rows.append({
            "model": name,
            "role": "classifier",
            "n_train": len(y_tr),
            "n_test": len(X_te),
            "fit_s": round(fit_s, 4),
            "predict_s": round(pred_s, 6),
            "predict_us_per_instance": round(1e6 * pred_s / len(X_te), 2),
            "notes": f"median of {BENCH_REPEATS} runs, sklearn",
        })
    return rows


if __name__ == "__main__":
    main()
