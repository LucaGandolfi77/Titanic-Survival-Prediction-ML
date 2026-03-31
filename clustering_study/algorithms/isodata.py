"""
ISODATA Implementation
========================
Full ISODATA with split, merge, and discard operations.

Algorithm outline (Ball & Hall, 1965):
1. Initialise K centres (random or k-means++ warm-start).
2. Assign each point to nearest centre.
3. Discard clusters smaller than θ_N.
4. Compute intra-cluster stats (std per feature, avg distance).
5. Split:  if any feature σ > θ_S and cluster is large enough.
6. Merge:  if distance between any two centres < θ_C.
7. Update centres.  Repeat until max_iter or no changes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from config import CFG


class ISODATA:
    """ISODATA clustering with split/merge/discard operations."""

    def __init__(
        self,
        k_init: int = CFG.ISODATA_K_INIT,
        theta_n: int = CFG.ISODATA_THETA_N,
        theta_s: float = CFG.ISODATA_THETA_S,
        theta_c: float = CFG.ISODATA_THETA_C,
        max_merge: int = CFG.ISODATA_MAX_MERGE,
        max_iter: int = CFG.ISODATA_MAX_ITER,
        random_state: int = 42,
    ):
        self.k_init = k_init
        self.theta_n = theta_n
        self.theta_s = theta_s
        self.theta_c = theta_c
        self.max_merge = max_merge
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.history_: List[Dict] = []

    def fit(self, X: np.ndarray) -> "ISODATA":
        n, d = X.shape
        k = min(self.k_init, n)

        # initialise with random sample
        idx = self.rng.choice(n, size=k, replace=False)
        centroids = X[idx].copy()

        for iteration in range(self.max_iter):
            labels, centroids, k = self._assign(X, centroids)

            # discard small clusters
            centroids, labels, k, n_discarded = self._discard(
                X, centroids, labels, k,
            )
            if k == 0:
                idx = self.rng.choice(n, size=1, replace=False)
                centroids = X[idx].copy()
                k = 1
                labels = np.zeros(n, dtype=int)

            # split
            centroids, labels, k, n_splits = self._split(X, centroids, labels, k)

            # merge
            centroids, labels, k, n_merges = self._merge(X, centroids, labels, k)

            # recompute centroids
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)

            self.history_.append({
                "iteration": iteration,
                "k": k,
                "splits": n_splits,
                "merges": n_merges,
                "discarded": n_discarded,
            })

            if n_splits == 0 and n_merges == 0 and n_discarded == 0 and iteration > 0:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.n_clusters_ = k
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(
            X[:, None, :] - self.centroids_[None, :, :], axis=2,
        )
        return np.argmin(dists, axis=1)

    # ── internal operations ───────────────────────────────────────

    @staticmethod
    def _assign(
        X: np.ndarray, centroids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        dists = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2,
        )
        labels = np.argmin(dists, axis=1)
        return labels, centroids, len(centroids)

    def _discard(
        self, X: np.ndarray, centroids: np.ndarray,
        labels: np.ndarray, k: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        keep = []
        for c in range(k):
            if np.sum(labels == c) >= self.theta_n:
                keep.append(c)
        n_discarded = k - len(keep)
        if n_discarded == 0:
            return centroids, labels, k, 0
        centroids = centroids[keep]
        labels, centroids, k = self._assign(X, centroids)
        return centroids, labels, k, n_discarded

    def _split(
        self, X: np.ndarray, centroids: np.ndarray,
        labels: np.ndarray, k: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        new_centroids = list(centroids)
        n_splits = 0
        clusters_to_check = list(range(k))

        for c in clusters_to_check:
            members = X[labels == c]
            if len(members) < 2 * self.theta_n:
                continue
            stds = np.std(members, axis=0)
            max_std_idx = int(np.argmax(stds))
            if stds[max_std_idx] > self.theta_s:
                offset = np.zeros(centroids.shape[1])
                offset[max_std_idx] = stds[max_std_idx] * 0.5
                c1 = new_centroids[c] + offset
                c2 = new_centroids[c] - offset
                new_centroids[c] = c1
                new_centroids.append(c2)
                n_splits += 1

        if n_splits > 0:
            centroids = np.array(new_centroids)
            labels, centroids, k = self._assign(X, centroids)
        return centroids, labels, k, n_splits

    def _merge(
        self, X: np.ndarray, centroids: np.ndarray,
        labels: np.ndarray, k: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        if k < 2:
            return centroids, labels, k, 0

        # find close pairs
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                d = float(np.linalg.norm(centroids[i] - centroids[j]))
                if d < self.theta_c:
                    pairs.append((d, i, j))
        pairs.sort()

        merged = set()
        n_merges = 0
        new_centroids = list(centroids)
        remove_idx = set()

        for _, i, j in pairs:
            if n_merges >= self.max_merge:
                break
            if i in merged or j in merged:
                continue
            ni = np.sum(labels == i)
            nj = np.sum(labels == j)
            total = ni + nj
            if total == 0:
                continue
            merged_centroid = (centroids[i] * ni + centroids[j] * nj) / total
            new_centroids[i] = merged_centroid
            remove_idx.add(j)
            merged.add(i)
            merged.add(j)
            n_merges += 1

        if n_merges > 0:
            keep = [c for c in range(k) if c not in remove_idx]
            centroids = np.array([new_centroids[c] for c in keep])
            labels, centroids, k = self._assign(X, centroids)

        return centroids, labels, k, n_merges


def build_isodata(random_state: int = 42, **kwargs) -> ISODATA:
    return ISODATA(random_state=random_state, **kwargs)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=1.0,
                           random_state=42)
    iso = ISODATA(k_init=10, theta_s=1.2, theta_c=2.0, random_state=42)
    iso.fit(X)
    print(f"ISODATA found {iso.n_clusters_} clusters")
    for h in iso.history_:
        print(f"  iter={h['iteration']}  k={h['k']}  "
              f"splits={h['splits']}  merges={h['merges']}  "
              f"discarded={h['discarded']}")
