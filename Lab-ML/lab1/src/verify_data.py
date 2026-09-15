"""Section 2.6 verification gate.

Asserts that every source dataset matches the inventory in the plan
(``Lab-ML/LAB1_AGENT_TODO.md`` section 1.1) and prints a summary.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arff_utils import (  # noqa: E402
    SOURCE_DATA_DIR,
    encode_labels,
    load_arff,
)

EXPECTED = {
    # filename: (n_rows, {class: count})
    "circletrain.arff": (100, {"c": 53, "q": 47}),
    "circletest.arff": (100, {"c": 60, "q": 40}),
    "circleall.arff": (2601, {"c": 1251, "q": 1350}),
    "Bigtest1_104.arff": (
        6024,
        {"0": 557, "1": 525, "2": 566, "3": 739, "4": 601,
         "5": 619, "6": 599, "7": 533, "8": 611, "9": 674},
    ),
    "Bigtest2_104.arff": (5010, {str(d): 501 for d in range(10)}),
}


def main() -> int:
    failures = []
    for filename, (n_expected, class_expected) in EXPECTED.items():
        path = os.path.join(SOURCE_DATA_DIR, filename)
        X, y, attrs, classes, relation = load_arff(path)

        ok_rows = len(y) == n_expected
        if not ok_rows:
            failures.append(f"{filename}: {len(y)} rows, expected {n_expected}")

        values, counts = np.unique(y, return_counts=True)
        actual = {str(v): int(c) for v, c in zip(values, counts)}
        ok_classes = actual == class_expected
        if not ok_classes:
            failures.append(f"{filename}: class counts {actual} != {class_expected}")

        print(f"\n=== {filename} ===")
        print(f"  relation : {relation}")
        print(f"  shape    : {X.shape}   attrs[:4]={attrs[:4]}"
              f"{' ...' if len(attrs) > 4 else ''}")
        print(f"  classes  : {classes}")
        print(f"  labels   : {actual}")
        print(f"  dtype    : {X.dtype}, finite={np.isfinite(X).all()}")
        print(f"  min/max  : {X.min(axis=0)[:4]} / {X.max(axis=0)[:4]}")
        if "circle" in filename:
            print(f"  x range  : {X[:, 0].min():.4f} .. {X[:, 0].max():.4f}")
            print(f"  y range  : {X[:, 1].min():.4f} .. {X[:, 1].max():.4f}")
            uniq = np.unique(X[:, 0])
            if len(uniq) > 1:
                step = np.diff(uniq)
                print(f"  x step   : min={step.min():.4f} max={step.max():.4f}")
            y_int = encode_labels(y)
            print(f"  int labels (c=0, q=1): counts={np.bincount(y_int)}")
        print(f"  {'OK' if ok_rows and ok_classes else 'MISMATCH'}")

    print("\n" + "=" * 60)
    if failures:
        print("VERIFICATION GATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFICATION GATE PASSED: all datasets match the inventory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
