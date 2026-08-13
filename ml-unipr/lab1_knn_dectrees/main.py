"""main.py — Run all lab 1 exercises (kNN & Decision Trees) in sequence."""

from __future__ import annotations

import os

import exercise1_knn
import exercise1b_repr
import exercise2_digits
import exercise2b_overfit


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    exercise1_knn.run()
    exercise1b_repr.run()
    exercise2_digits.run()
    exercise2b_overfit.run()

    print("\n" + "=" * 64)
    print("All exercises complete.  See outputs/ for plots and CSVs.")
    print("=" * 64)


if __name__ == "__main__":
    main()
