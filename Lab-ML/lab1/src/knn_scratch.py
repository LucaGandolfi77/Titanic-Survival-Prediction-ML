"""From-scratch k Nearest Neighbors classifier (plan section 3.1).

The only correctness proof that matters is that this implementation agrees
point-for-point with scikit-learn's ``KNeighborsClassifier`` - see
:func:`self_test` and the ``__main__`` block.

Label encoding (documented once, used everywhere):
    c (inside the circle / inner region) -> 0
    q (inside the square ring)           -> 1
so **q is the "positive" class**.  Note that class 0 is also the tie-break
winner, which is why ties are counted and reported rather than hidden.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

TIE_BREAK_CLASS = 0  # 'c'


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix, vectorised.

    Parameters
    ----------
    A : (n, d), B : (m, d)

    Returns
    -------
    (n, m) array with ``D[i, j] = ||A[i] - B[j]||``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A[None, :]
    if B.ndim == 1:
        B = B[None, :]
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimension mismatch: {A.shape} vs {B.shape}")
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)  # guard against tiny negative values
    return np.sqrt(d2)


def _majority_vote(neighbour_labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Vectorised majority vote over the neighbour axis.

    ``neighbour_labels`` has shape (n_queries, k).  Ties are broken towards
    :data:`TIE_BREAK_CLASS` and the number of tied queries is returned so it
    can be reported.
    """
    n_queries, k = neighbour_labels.shape
    n_classes = int(neighbour_labels.max()) + 1 if k else 1
    counts = np.zeros((n_queries, n_classes), dtype=int)
    rows = np.repeat(np.arange(n_queries), k)
    np.add.at(counts, (rows, neighbour_labels.ravel()), 1)

    # Deterministic argmax: numpy returns the first maximum, and the classes are
    # ordered ascending, so equal counts resolve to the smallest label.
    pred = np.argmax(counts, axis=1)

    if k % 2 == 0:
        top = np.sort(counts, axis=1)[:, -2:]
        tied = int(np.sum(top[:, 0] == top[:, 1]))
    else:
        # With odd k and two classes a tie is impossible; keep the counter for
        # the general (possibly even k) case.
        top = np.sort(counts, axis=1)[:, -2:]
        tied = int(np.sum(top[:, 0] == top[:, 1]))
    return pred.astype(int), tied


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    return_info: bool = False,
):
    """Classify ``X_query`` by majority vote of the k nearest training samples."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(y_train):
        raise ValueError(f"k={k} exceeds the training set size {len(y_train)}")

    D = euclidean(X_query, X_train)                       # (n_query, n_train)
    # Indices of the k smallest distances per row, sorted by distance so that
    # the vote is deterministic even for exactly equal distances.
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    neighbour_labels = np.asarray(y_train, dtype=int)[idx]
    pred, n_tied = _majority_vote(neighbour_labels)

    if return_info:
        info: Dict[str, object] = {
            "n_tied": n_tied,
            "k": k,
            "mean_nn_distance": float(D.min(axis=1).mean()),
        }
        return pred, info
    return pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    return float(np.mean(y_true == y_pred))


# --------------------------------------------------------------------------
# Leave-one-out and k-fold cross validation
# --------------------------------------------------------------------------


def loo_predictions(
    X: np.ndarray, y: np.ndarray, k: int
) -> np.ndarray:
    """Predict every sample from the remaining n-1 samples (leave-one-out)."""
    D = euclidean(X, X)
    np.fill_diagonal(D, np.inf)          # never let a point vote for itself
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    neighbour_labels = np.asarray(y, dtype=int)[idx]
    pred, _ = _majority_vote(neighbour_labels)
    return pred


def loo_accuracy(X: np.ndarray, y: np.ndarray, k: int) -> float:
    return accuracy(y, loo_predictions(X, y, k))


def stratified_folds(y: np.ndarray, n_folds: int, seed: int = 0):
    """Return a list of (train_idx, test_idx) stratified folds."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        # distribute the class round-robin over the folds, keeping them even
        for i, sample in enumerate(idx):
            folds[i % n_folds].append(int(sample))
    out = []
    all_idx = set(range(len(y)))
    for f in folds:
        test_idx = np.array(sorted(f), dtype=int)
        train_idx = np.array(sorted(all_idx - set(f)), dtype=int)
        out.append((train_idx, test_idx))
    return out


def cv_accuracy(
    X: np.ndarray, y: np.ndarray, k: int, n_folds: int = 5, seed: int = 0
) -> Tuple[float, float, list]:
    """Stratified k-fold CV accuracy: returns (mean, std, per-fold accuracies)."""
    accs = []
    per_fold = []
    for train_idx, test_idx in stratified_folds(y, n_folds, seed=seed):
        pred = knn_predict(X[train_idx], y[train_idx], X[test_idx], k)
        acc = accuracy(y[test_idx], pred)
        accs.append(acc)
        per_fold.append(
            {"n_train": len(train_idx), "n_test": len(test_idx), "accuracy": acc}
        )
    return float(np.mean(accs)), float(np.std(accs)), per_fold


def cv_accuracy_multi_k(
    X: np.ndarray,
    y: np.ndarray,
    ks,
    n_folds: int = 5,
    seed: int = 0,
) -> Dict[int, Tuple[float, float, list]]:
    """Same as :func:`cv_accuracy` but reuses each fold's distance matrix.

    Returns ``{k: (mean, std, per_fold)}``.  Recomputing the pairwise distances
    once per fold instead of once per (fold, k) is the only reason this function
    exists - the numbers are identical to looping over :func:`cv_accuracy`.
    """
    folds = stratified_folds(y, n_folds, seed=seed)
    out: Dict[int, Tuple[float, float, list]] = {int(k): ([], [], []) for k in ks}  # type: ignore[assignment]
    for train_idx, test_idx in folds:
        y_train, y_test = y[train_idx], y[test_idx]
        D = euclidean(X[test_idx], X[train_idx])          # (n_test, n_train)
        order = np.argsort(D, axis=1, kind="stable")
        for k in ks:
            k = int(k)
            # Inside a CV fold the training set is smaller than the full set, so
            # a large k must be clamped to the largest usable odd value.
            k_eff = min(k, len(train_idx))
            if k_eff % 2 == 0:
                k_eff -= 1
            if k_eff < 1:
                continue
            nb = np.asarray(y_train, dtype=int)[order[:, :k_eff]]
            pred, _ = _majority_vote(nb)
            acc = accuracy(y_test, pred)
            out[k][0].append(acc)
            out[k][2].append(
                {
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "k_effective": k_eff,
                    "accuracy": acc,
                }
            )
    return {
        k: (float(np.mean(v[0])), float(np.std(v[0])), v[2]) for k, v in out.items()
    }


# --------------------------------------------------------------------------
# Correctness proof against scikit-learn
# --------------------------------------------------------------------------


def self_test(verbose: bool = True) -> bool:
    """Assert point-for-point agreement with scikit-learn on real lab data."""
    from sklearn.neighbors import KNeighborsClassifier

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arff_utils import load_circle

    X_train, y_train, _ = load_circle("train")
    X_test, y_test, _ = load_circle("test")

    all_ok = True
    for k in (1, 3, 5, 7, 15, 31):
        mine = knn_predict(X_train, y_train, X_test, k)
        sk = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(
            X_train, y_train
        ).predict(X_test)
        agree = int(np.sum(mine == sk))
        ok = agree == len(y_test)
        all_ok &= ok
        if verbose:
            print(
                f"  k={k:>2}: mine={accuracy(y_test, mine):.4f} "
                f"sklearn={accuracy(y_test, sk):.4f} "
                f"agree={agree}/{len(y_test)} {'OK' if ok else 'MISMATCH'}"
            )

    # Also check the distance function against scipy.
    from scipy.spatial.distance import cdist

    D_mine = euclidean(X_train, X_test)
    D_scipy = cdist(X_train, X_test, metric="euclidean")
    d_ok = np.allclose(D_mine, D_scipy, atol=1e-10)
    all_ok &= d_ok
    if verbose:
        print(f"  euclidean() vs scipy.cdist: max|diff|={np.abs(D_mine - D_scipy).max():.2e} "
              f"{'OK' if d_ok else 'MISMATCH'}")

    # LOO sanity: k=1 on the training set must be a perfect fit (each point's
    # nearest neighbour excluding itself is not necessarily itself, so this is
    # NOT guaranteed to be 100% - report the value instead of asserting it).
    loo1 = loo_accuracy(X_train, y_train, 1)
    if verbose:
        print(f"  LOO k=1 accuracy on circletrain: {loo1:.4f}")

    return all_ok


if __name__ == "__main__":
    print("Self-test: from-scratch kNN vs scikit-learn")
    ok = self_test()
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    raise SystemExit(0 if ok else 1)
