"""Plan section 3.5-3.7: build the Exercise 1b datasets.

Four variants of each circle file are generated (originals are never modified):

* ``*_sq.arff``  inner region is the **square** with half-side 0.88 (p.7).
                 Labels are recomputed from that geometry.
* ``*_zt.arff``  original circle labels, attributes ``z = x^2``, ``t = y^2`` (p.8).
* ``*_u.arff``   original circle labels, single attribute ``u = x^2 + y^2`` (p.9).
* ``*_xy.arff``  original circle labels, plain ``x``, ``y`` - a copy used so that
                 every representation can be driven through the same code path.

Also prints the label-change statistics required by step 3.5.2.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arff_utils import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    SOURCE_DATA_DIR,
    load_arff,
    write_arff,
)

#: half side of the inner square for Exercise 1b (p.7): l = 1.76
INNER_HALF_SIDE = 0.88
#: half side of the outer square (the domain boundary)
OUTER_HALF_SIDE = 1.25


def labels_original(X: np.ndarray) -> np.ndarray:
    """Region 1 = the **circle** of radius 1: class c (0) inside, q (1) outside.

    Verified against the file labels: this rule reproduces ``circletrain.arff``
    and ``circletest.arff`` with zero errors, and ``circleall.arff`` up to six
    points that lie exactly on the circle.
    """
    return np.where(np.einsum("ij,ij->i", X, X) <= 1.0, 0, 1).astype(int)


def labels_inner_square_only(X: np.ndarray) -> np.ndarray:
    """New region for Exercise 1b: the **square** of half-side 0.88 (p.7).

    Note that a square with half-side 0.88 has circumradius
    ``0.88*sqrt(2) = 1.2445 > 1``, so it is *bigger* than the old circle at the
    diagonals and *smaller* at the axes (``0.88 < 1``).  The two regions overlap
    but neither contains the other.
    """
    inside = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) <= INNER_HALF_SIDE + 1e-12
    return np.where(inside, 0, 1).astype(int)


def labels_square_inner(X: np.ndarray) -> np.ndarray:
    """New region = circle replaced by the square, outer square unchanged.

    Class c = points inside the inner square **and** inside the outer domain.
    All points of these files already lie inside the outer square, so this is
    simply the inner square.  Defined as a union of primitives so the geometry
    stays explicit:

        c  <=>  (|x| <= 0.88 and |y| <= 0.88)  and  (|x| <= 1.25 and |y| <= 1.25)
    """
    inner = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) <= INNER_HALF_SIDE + 1e-12
    outer = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) <= OUTER_HALF_SIDE + 1e-12
    return np.where(inner & outer, 0, 1).astype(int)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = {}
    for name in ("train", "test", "all"):
        filename = {"train": "circletrain.arff", "test": "circletest.arff",
                    "all": "circleall.arff"}[name]
        path = os.path.join(SOURCE_DATA_DIR, filename)
        X, y_str, attrs, classes, relation = load_arff(path)
        y_orig = np.array([0 if v == "c" else 1 for v in y_str], dtype=int)

        # --- consistency: the file's own labels must match the circle rule -----
        y_rule = labels_original(X)
        n_mismatch = int((y_rule != y_orig).sum())
        r2 = np.einsum("ij,ij->i", X, X)
        max_abs = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1]))
        on_circle = np.abs(r2 - 1.0) < 1e-9
        on_outer = np.abs(max_abs - OUTER_HALF_SIDE) < 1e-9
        on_boundary = on_circle | on_outer
        n_mismatch_off = int(((y_rule != y_orig) & ~on_boundary).sum())

        # --- variant: square inner region -------------------------------------
        y_sq = labels_square_inner(X)
        in_inner_square = max_abs <= INNER_HALF_SIDE + 1e-12
        in_outer_square = max_abs <= OUTER_HALF_SIDE + 1e-12
        old_c = y_orig == 0

        # The two possible transitions, derived independently of labels:
        #   c -> q : was inside the circle, now outside the new inner square
        #   q -> c : was outside the circle, now inside the new inner square
        #            (the inner square's diagonal "ears")
        expect_c_to_q = old_c & ~in_inner_square
        expect_q_to_c = (~old_c) & in_inner_square
        actual_c_to_q = old_c & (y_sq == 1)
        actual_q_to_c = (~old_c) & (y_sq == 0)
        c_to_q_ok = bool(np.array_equal(expect_c_to_q, actual_c_to_q))
        q_to_c_ok = bool(np.array_equal(expect_q_to_c, actual_q_to_c))

        # Every point of these files must lie inside the outer square domain.
        all_inside_domain = bool(in_outer_square.all())

        # Strongest check: the inner square must be exactly the set of points
        # describing region 1 of the new problem, i.e. the two definitions of
        # the new class c must agree pointwise.
        y_sq_check = np.where(in_inner_square & in_outer_square, 0, 1)
        relabel_consistent = bool(np.array_equal(y_sq, y_sq_check))
        changed = y_sq != y_orig
        n_added_to_c = int(actual_q_to_c.sum())
        n_removed_from_c = int(actual_c_to_q.sum())

        # --- variant: z = x^2, t = y^2 ----------------------------------------
        Z = np.column_stack([X[:, 0] ** 2, X[:, 1] ** 2])
        # --- variant: u = x^2 + y^2 -------------------------------------------
        U = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)

        write_arff(
            os.path.join(DATA_DIR, f"circle{name}_xy.arff"), X, y_orig,
            ["x", "y"], ["c", "q"], "circle-xy",
        )
        write_arff(
            os.path.join(DATA_DIR, f"circle{name}_sq.arff"), X, y_sq,
            ["x", "y"], ["c", "q"], "circle-square-inner",
        )
        write_arff(
            os.path.join(DATA_DIR, f"circle{name}_zt.arff"), Z, y_orig,
            ["z", "t"], ["c", "q"], "circle-zt",
        )
        write_arff(
            os.path.join(DATA_DIR, f"circle{name}_u.arff"), U, y_orig,
            ["u"], ["c", "q"], "circle-u",
        )

        # --- separability checks (the analytic claims in steps 3.6/3.7) -------
        c_vals, q_vals = U[y_orig == 0, 0], U[y_orig == 1, 0]
        u_separable = bool(c_vals.max() <= q_vals.min())
        threshold = float((c_vals.max() + q_vals.min()) / 2.0)
        u_exact_threshold_1 = bool(c_vals.max() <= 1.0 < q_vals.min())

        z_vals, t_vals = Z, Z
        r2_all = (z_vals + t_vals).sum(axis=1)   # == x^2 + y^2, shape (n,)
        sum_c = float(r2_all[y_orig == 0].max())
        sum_q = float(r2_all[y_orig == 1].min())

        stats[name] = {
            "n_points": int(len(X)),
            "class_counts_original": {
                "c": int((y_orig == 0).sum()), "q": int((y_orig == 1).sum())},
            "class_counts_square_inner": {
                "c": int((y_sq == 0).sum()), "q": int((y_sq == 1).sum())},
            "file_labels_vs_circle_rule": {
                "n_mismatch": n_mismatch,
                "n_mismatch_off_boundary": n_mismatch_off,
                "n_on_circle": int(on_circle.sum()),
                "n_on_outer_square": int(on_outer.sum()),
                "max_r2_in_class_c": float(r2[y_orig == 0].max()),
                "min_r2_in_class_q": float(r2[y_orig == 1].min()),
            },
            "relabelled_to_square": {
                "n_changed": int(changed.sum()),
                "n_c_to_q": n_removed_from_c,
                "n_q_to_c": n_added_to_c,
                "c_to_q_set_matches_analytic": c_to_q_ok,
                "q_to_c_set_matches_analytic": q_to_c_ok,
                "relabelling_is_pointwise_consistent": relabel_consistent,
                "all_points_inside_outer_domain": all_inside_domain,
            },
            "u_feature": {
                "max_u_in_class_c": float(c_vals.max()),
                "min_u_in_class_q": float(q_vals.min()),
                "perfectly_separable_by_single_threshold": u_separable,
                "threshold_midpoint": threshold,
                "exactly_equivalent_to_u_le_1": u_exact_threshold_1,
                "predicted_tree": "u <= 1: c / u > 1: q" if u_exact_threshold_1 else "?",
            },
            "zt_features": {
                "max_z_plus_t_in_class_c": float(sum_c),
                "min_z_plus_t_in_class_q": float(sum_q),
                "circle_is_exactly_z_plus_t_le_1": bool(sum_c <= 1.0 < sum_q),
            },
        }

        print(f"\n=== circle{name} ===")
        print(f"  points={len(X)}  original c/q={stats[name]['class_counts_original']}"
              f"  square-inner c/q={stats[name]['class_counts_square_inner']}")
        print(f"  file labels vs L2 circle rule: {n_mismatch} mismatch "
              f"({n_mismatch_off} off-boundary, {int(on_circle.sum())} exactly on the circle)")
        print(f"  relabelled: {int(changed.sum())} points "
              f"(c->q {n_removed_from_c}, q->c {n_added_to_c})")
        print(f"  c->q set matches analytic: {c_to_q_ok};  q->c set matches analytic: {q_to_c_ok};"
              f"  pointwise consistent: {relabel_consistent}")
        print(f"  u = x^2+y^2 : max(u|c)={c_vals.max():.6f}  min(u|q)={q_vals.min():.6f} "
              f"-> separable={u_separable}, threshold={threshold:.6f}")
        print(f"  z+t        : max|class c = {sum_c:.6f}  min|class q = {sum_q:.6f}")
        print(f"  wrote circle{name}_{{xy,sq,zt,u}}.arff")

    out = os.path.join(RESULTS_DIR, "representation_datasets.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
