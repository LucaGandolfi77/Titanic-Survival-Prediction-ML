"""
Adaptive Clustering Framework
================================
Dynamic split/merge driven by silhouette per cluster, centroid
separation, and bootstrap stability.

Strategy:
1. Start with K-Means at k_init.
2. For each cluster, compute its silhouette coefficient.
3. SPLIT clusters with silhouette < split_threshold
   (indicates poor cohesion — cluster too heterogeneous).
4. MERGE pairs of clusters whose centroid distance < merge_threshold
   (indicates poor separation — clusters too similar).
5. Re-evaluate stability across bootstrap resamples.
6. Repeat until all clusters satisfy thresholds or max_iter reached.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from config import CFG


class AdaptiveClustering:
    """Adaptive K selection via iterative split/merge."""

    def __init__(
        self,
        k_init: int = CFG.DEFAULT_K,
        k_min: int = CFG.ADAPT_K_MIN,
        k_max: int = CFG.ADAPT_K_MAX,
        split_silhouette: float = CFG.ADAPT_SPLIT_SILHOUETTE,
        merge_distance: float = CFG.ADAPT_MERGE_DISTANCE,
        stability_threshold: float = CFG.ADAPT_STABILITY_THRESHOLD,
        max_iter: int = CFG.ADAPT_MAX_ITER,
        n_bootstrap: int = CFG.ADAPT_N_BOOTSTRAP,
        random_state: int = 42,
    ):
        self.k_init = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.split_silhouette = split_silhouette
        self.merge_distance = merge_distance
        self.stability_threshold = stability_threshold
        self.max_iter = max_iter
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.history_: List[Dict] = []

    def fit(self, X: np.ndarray) -> "AdaptiveClustering":
        n, d = X.shape
        k = min(self.k_init, n // 2, self.k_max)
        k = max(k, self.k_min)

        # initial clustering
        centroids, labels = self._run_kmeans(X, k)

        for iteration in range(self.max_iter):
            k_before = len(centroids)

            # split phase
            centroids, labels, n_splits = self._split_phase(X, centroids, labels)

            # merge phase
            centroids, labels, n_merges = self._merge_phase(X, centroids, labels)

            k_after = len(centroids)
            global_sil = (
                float(silhouette_score(X, labels))
                if len(np.unique(labels)) >= 2 else -1.0
            )

            self.history_.append({
                "iteration": iteration,
                "k_before": k_before,
                "k_after": k_after,
                "splits": n_splits,
                "merges": n_merges,
                "silhouette": global_sil,
            })

            if n_splits == 0 and n_merges == 0:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.n_clusters_ = len(centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(
            X[:, None, :] - self.centroids_[None, :, :], axis=2,
        )
        return np.argmin(dists, axis=1)

    # ── split / merge ─────────────────────────────────────────────

    def _split_phase(
        self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        k = len(centroids)
        if k >= self.k_max:
            return centroids, labels, 0

        n_unique = len(np.unique(labels))
        if n_unique < 2:
            return centroids, labels, 0

        sil_samples = silhouette_samples(X, labels)
        new_centroids = list(centroids)
        n_splits = 0

        for c in range(k):
            if len(new_centroids) >= self.k_max:
                break
            mask = labels == c
            cluster_sil = float(np.mean(sil_samples[mask])) if mask.sum() > 0 else 0.0
            n_members = int(mask.sum())

            if cluster_sil < self.split_silhouette and n_members >= 4:
                members = X[mask]
                stds = np.std(members, axis=0)
                max_dim = int(np.argmax(stds))
                offset = np.zeros(centroids.shape[1])
                offset[max_dim] = stds[max_dim] * 0.5
                c1 = new_centroids[c] + offset
                c2 = new_centroids[c] - offset
                new_centroids[c] = c1
                new_centroids.append(c2)
                n_splits += 1

        if n_splits > 0:
            centroids = np.array(new_centroids)
            labels = self._assign(X, centroids)

        return centroids, labels, n_splits

    def _merge_phase(
        self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        k = len(centroids)
        if k <= self.k_min:
            return centroids, labels, 0

        # find close pairs
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                d = float(np.linalg.norm(centroids[i] - centroids[j]))
                if d < self.merge_distance:
                    pairs.append((d, i, j))
        pairs.sort()

        merged_set: set = set()
        new_centroids = list(centroids)
        remove_idx: set = set()
        n_merges = 0

        for _, i, j in pairs:
            if len(new_centroids) - len(remove_idx) <= self.k_min:
                break
            if i in merged_set or j in merged_set:
                continue
            ni = int(np.sum(labels == i))
            nj = int(np.sum(labels == j))
            total = ni + nj
            if total == 0:
                continue
            merged_c = (centroids[i] * ni + centroids[j] * nj) / total
            new_centroids[i] = merged_c
            remove_idx.add(j)
            merged_set.update({i, j})
            n_merges += 1

        if n_merges > 0:
            keep = [c for c in range(k) if c not in remove_idx]
            centroids = np.array([new_centroids[c] for c in keep])
            labels = self._assign(X, centroids)

        return centroids, labels, n_merges

    def _run_kmeans(
        self, X: np.ndarray, k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state)
        labels = km.fit_predict(X)
        return km.cluster_centers_.copy(), labels

    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2,
        )
        return np.argmin(dists, axis=1)

    def compute_stability(self, X: np.ndarray) -> float:
        """Compute bootstrap stability of current partition."""
        if self.centroids_ is None:
            return 0.0
        from sklearn.metrics import adjusted_rand_score

        base_labels = self.labels_
        scores = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            km = KMeans(n_clusters=self.n_clusters_, n_init=3,
                        random_state=int(self.rng.integers(10000)))
            boot_labels = km.fit_predict(X_boot)
            # compare on the bootstrap sample
            ari = adjusted_rand_score(base_labels[idx], boot_labels)
            scores.append(ari)
        return float(np.mean(scores))


def build_adaptive(
    n_clusters: int = CFG.DEFAULT_K,
    random_state: int = 42,
    **kwargs,
) -> AdaptiveClustering:
    return AdaptiveClustering(k_init=n_clusters, random_state=random_state, **kwargs)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=1.0,
                           random_state=42)
    ac = AdaptiveClustering(k_init=2, split_silhouette=0.3,
                            merge_distance=1.0, random_state=42)
    ac.fit(X)
    print(f"Adaptive found {ac.n_clusters_} clusters")
    print(f"Stability: {ac.compute_stability(X):.3f}")
    for h in ac.history_:
        print(f"  iter={h['iteration']}  k={h['k_after']}  "
              f"splits={h['splits']}  merges={h['merges']}  "
              f"sil={h['silhouette']:.3f}")
