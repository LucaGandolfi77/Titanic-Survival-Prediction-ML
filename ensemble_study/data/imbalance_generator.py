"""
Imbalance Generator
====================
Create class-imbalanced versions of datasets by undersampling the
majority class.  Supports arbitrary minority:majority ratios.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.utils import resample

from config import parse_ratio


def generate_imbalanced(
    X: np.ndarray,
    y: np.ndarray,
    ratio_str: str,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Undersample majority to achieve target minority:majority ratio.

    For multi-class problems, the smallest class is treated as minority
    and all other classes are undersampled proportionally.

    Returns (X_imb, y_imb, metadata).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    ratio = parse_ratio(ratio_str)
    classes, counts = np.unique(y, return_counts=True)

    if ratio >= 1.0:
        return X.copy(), y.copy(), _meta(ratio_str, len(y), len(y))

    minority_idx = int(np.argmin(counts))
    minority_class = classes[minority_idx]
    n_minority = counts[minority_idx]

    # Target majority count so that minority/majority = ratio
    target_maj = int(n_minority / ratio)

    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if cls == minority_class:
            indices.append(cls_idx)
        else:
            n_keep = min(target_maj, len(cls_idx))
            chosen = rng.choice(cls_idx, size=n_keep, replace=False)
            indices.append(chosen)

    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)

    return X[all_idx].copy(), y[all_idx].copy(), _meta(ratio_str, len(all_idx), len(y))


def _meta(ratio_str: str, n_after: int, n_before: int) -> Dict:
    return {
        "type": "imbalance",
        "ratio": ratio_str,
        "n_samples_after": n_after,
        "n_samples_before": n_before,
    }


if __name__ == "__main__":
    from data.loaders import get_dataset_by_name

    X, y, _ = get_dataset_by_name("breast_cancer")
    for r in ["1:1", "1:2", "1:5", "1:10"]:
        X_i, y_i, meta = generate_imbalanced(X, y, r, np.random.default_rng(42))
        classes, counts = np.unique(y_i, return_counts=True)
        print(f"ratio={r:5s}  n={len(y_i)}  dist={dict(zip(classes, counts))}")
