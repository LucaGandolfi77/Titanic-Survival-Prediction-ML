"""Tests for Adaptive Clustering framework."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from algorithms.adaptive_clustering import AdaptiveClustering, build_adaptive


class TestAdaptiveClustering(unittest.TestCase):
    """Verify adaptive split/merge and stability."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=400, centers=4, cluster_std=0.7, random_state=42,
        )

    # ── basics ─────────────────────────────────────────────────

    def test_build_adaptive(self):
        ac = build_adaptive(n_clusters=3, random_state=42)
        self.assertIsInstance(ac, AdaptiveClustering)

    def test_fit_returns_self(self):
        ac = AdaptiveClustering(k_init=3, random_state=42)
        self.assertIs(ac.fit(self.X), ac)

    def test_fit_assigns_attributes(self):
        ac = AdaptiveClustering(k_init=3, random_state=42)
        ac.fit(self.X)
        self.assertIsNotNone(ac.centroids_)
        self.assertIsNotNone(ac.labels_)
        self.assertGreater(ac.n_clusters_, 0)
        self.assertEqual(len(ac.labels_), len(self.X))

    def test_predict_shape(self):
        ac = AdaptiveClustering(k_init=3, random_state=42)
        ac.fit(self.X)
        labels = ac.predict(self.X)
        self.assertEqual(labels.shape, (len(self.X),))

    # ── split phase ────────────────────────────────────────────

    def test_split_triggered_on_heterogeneous_start(self):
        ac = AdaptiveClustering(
            k_init=2, split_silhouette=0.8, merge_distance=0.01,
            max_iter=15, random_state=42,
        )
        ac.fit(self.X)
        split_total = sum(h["splits"] for h in ac.history_)
        self.assertGreater(split_total, 0, "Should split heterogeneous clusters")
        self.assertGreater(ac.n_clusters_, 2)

    # ── merge phase ────────────────────────────────────────────

    def test_merge_triggered_on_over_partitioned(self):
        ac = AdaptiveClustering(
            k_init=10, split_silhouette=-2.0, merge_distance=5.0,
            max_iter=10, random_state=42,
        )
        ac.fit(self.X)
        merge_total = sum(h["merges"] for h in ac.history_)
        self.assertGreater(merge_total, 0, "Should merge close clusters")
        self.assertLess(ac.n_clusters_, 10)

    # ── k bounds ───────────────────────────────────────────────

    def test_k_min_respected(self):
        ac = AdaptiveClustering(
            k_init=3, k_min=2, merge_distance=100.0,
            split_silhouette=-2.0, max_iter=5, random_state=42,
        )
        ac.fit(self.X)
        self.assertGreaterEqual(ac.n_clusters_, 2)

    def test_k_max_respected(self):
        ac = AdaptiveClustering(
            k_init=3, k_max=6, split_silhouette=0.9,
            merge_distance=0.01, max_iter=15, random_state=42,
        )
        ac.fit(self.X)
        self.assertLessEqual(ac.n_clusters_, 6)

    # ── history ────────────────────────────────────────────────

    def test_history_keys(self):
        ac = AdaptiveClustering(k_init=4, random_state=42)
        ac.fit(self.X)
        self.assertGreater(len(ac.history_), 0)
        keys = {"iteration", "k_before", "k_after", "splits", "merges", "silhouette"}
        self.assertTrue(keys.issubset(ac.history_[0].keys()))

    # ── stability ──────────────────────────────────────────────

    def test_compute_stability_range(self):
        ac = AdaptiveClustering(k_init=4, n_bootstrap=5, random_state=42)
        ac.fit(self.X)
        stab = ac.compute_stability(self.X)
        self.assertGreaterEqual(stab, -1.0)
        self.assertLessEqual(stab, 1.0)

    # ── reproducibility ───────────────────────────────────────

    def test_deterministic(self):
        ac1 = AdaptiveClustering(k_init=4, random_state=77)
        ac1.fit(self.X)
        ac2 = AdaptiveClustering(k_init=4, random_state=77)
        ac2.fit(self.X)
        np.testing.assert_array_equal(ac1.labels_, ac2.labels_)

    # ── edge case ──────────────────────────────────────────────

    def test_single_cluster_no_crash(self):
        X_tight = np.random.default_rng(0).normal(size=(50, 2)) * 0.01
        ac = AdaptiveClustering(k_init=2, max_iter=3, random_state=0)
        ac.fit(X_tight)
        self.assertGreater(ac.n_clusters_, 0)


if __name__ == "__main__":
    unittest.main()
