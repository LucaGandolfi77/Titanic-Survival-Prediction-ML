"""Tests for ISODATA clustering algorithm."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from algorithms.isodata import ISODATA, build_isodata


class TestISODATA(unittest.TestCase):
    """Verify ISODATA split, merge, discard, and convergence."""

    @classmethod
    def setUpClass(cls):
        cls.X_easy, cls.y_easy = make_blobs(
            n_samples=300, centers=3, cluster_std=0.5, random_state=42,
        )
        cls.X_overlap, cls.y_overlap = make_blobs(
            n_samples=400, centers=5, cluster_std=2.0, random_state=7,
        )

    # ── basic functionality ───────────────────────────────────

    def test_build_isodata(self):
        iso = build_isodata(random_state=42)
        self.assertIsInstance(iso, ISODATA)

    def test_fit_returns_self(self):
        iso = ISODATA(k_init=4, random_state=42)
        result = iso.fit(self.X_easy)
        self.assertIs(result, iso)

    def test_fit_assigns_attributes(self):
        iso = ISODATA(k_init=4, max_iter=20, random_state=42)
        iso.fit(self.X_easy)
        self.assertIsNotNone(iso.centroids_)
        self.assertIsNotNone(iso.labels_)
        self.assertGreater(iso.n_clusters_, 0)
        self.assertEqual(len(iso.labels_), len(self.X_easy))

    def test_predict_shape(self):
        iso = ISODATA(k_init=4, random_state=42)
        iso.fit(self.X_easy)
        labels = iso.predict(self.X_easy)
        self.assertEqual(labels.shape, (len(self.X_easy),))

    def test_labels_contiguous(self):
        iso = ISODATA(k_init=6, random_state=42)
        iso.fit(self.X_easy)
        unique = np.unique(iso.labels_)
        self.assertTrue(np.array_equal(unique, np.arange(iso.n_clusters_)))

    # ── split behaviour ────────────────────────────────────────

    def test_split_increases_k(self):
        iso = ISODATA(
            k_init=2, theta_s=0.3, theta_c=0.01, max_iter=10, random_state=42,
        )
        iso.fit(self.X_easy)
        split_total = sum(h["splits"] for h in iso.history_)
        self.assertGreater(split_total, 0, "ISODATA should split at least once")
        self.assertGreater(iso.n_clusters_, 2)

    # ── merge behaviour ────────────────────────────────────────

    def test_merge_reduces_k(self):
        iso = ISODATA(
            k_init=8, theta_s=100.0, theta_c=5.0, max_iter=15, random_state=42,
        )
        iso.fit(self.X_easy)
        merge_total = sum(h["merges"] for h in iso.history_)
        self.assertGreater(merge_total, 0, "ISODATA should merge at least once")

    # ── discard behaviour ──────────────────────────────────────

    def test_discard_removes_tiny_clusters(self):
        iso = ISODATA(
            k_init=10, theta_n=50, theta_s=100.0, theta_c=0.01,
            max_iter=5, random_state=42,
        )
        iso.fit(self.X_easy)
        discard_total = sum(h["discarded"] for h in iso.history_)
        self.assertGreater(discard_total, 0, "Should discard small clusters")

    # ── history tracking ───────────────────────────────────────

    def test_history_not_empty(self):
        iso = ISODATA(k_init=4, random_state=42)
        iso.fit(self.X_easy)
        self.assertGreater(len(iso.history_), 0)

    def test_history_keys(self):
        iso = ISODATA(k_init=4, random_state=42)
        iso.fit(self.X_easy)
        first = iso.history_[0]
        for key in ("iteration", "k", "splits", "merges", "discarded"):
            self.assertIn(key, first)

    # ── convergence ────────────────────────────────────────────

    def test_converges_within_max_iter(self):
        iso = ISODATA(k_init=5, max_iter=50, random_state=42)
        iso.fit(self.X_easy)
        self.assertLessEqual(len(iso.history_), 50)

    # ── reproducibility ───────────────────────────────────────

    def test_deterministic_with_same_seed(self):
        iso1 = ISODATA(k_init=5, random_state=99)
        iso1.fit(self.X_easy)
        iso2 = ISODATA(k_init=5, random_state=99)
        iso2.fit(self.X_easy)
        np.testing.assert_array_equal(iso1.labels_, iso2.labels_)
        np.testing.assert_array_almost_equal(iso1.centroids_, iso2.centroids_)

    # ── edge case: very small data ─────────────────────────────

    def test_tiny_dataset(self):
        X_tiny = np.array([[0, 0], [1, 1], [10, 10]])
        iso = ISODATA(k_init=2, theta_n=1, max_iter=5, random_state=42)
        iso.fit(X_tiny)
        self.assertGreater(iso.n_clusters_, 0)
        self.assertEqual(len(iso.labels_), 3)


if __name__ == "__main__":
    unittest.main()
