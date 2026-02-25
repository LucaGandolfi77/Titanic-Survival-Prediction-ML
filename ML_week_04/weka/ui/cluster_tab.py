"""
cluster_tab.py – Clustering panel (mirrors Weka Explorer ▸ Cluster).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from core.data_manager import DataManager
from core.evaluator import get_clusterers, run_clustering, ClusterResult
from ui.widgets import ScrolledLog, LabeledCombo, LabeledEntry


class ClusterTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self.results: list[ClusterResult] = []
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        # Algorithm selection
        alg_frame = ttk.LabelFrame(left, text="Clustering Algorithm")
        alg_frame.pack(fill=tk.X, padx=4, pady=4)
        alg_names = list(get_clusterers().keys())
        self.alg_combo = LabeledCombo(alg_frame, "Algorithm:", alg_names, width=22)
        self.alg_combo.pack(fill=tk.X, padx=6, pady=4)

        # Parameters
        param_frame = ttk.LabelFrame(left, text="Parameters")
        param_frame.pack(fill=tk.X, padx=4, pady=4)
        self.k_entry = LabeledEntry(param_frame, "Clusters (k):", default="3", width=6)
        self.k_entry.pack(fill=tk.X, padx=6, pady=3)
        self.eps_entry = LabeledEntry(param_frame, "DBSCAN eps:", default="0.5", width=6)
        self.eps_entry.pack(fill=tk.X, padx=6, pady=3)
        self.min_samples_entry = LabeledEntry(param_frame, "DBSCAN min_samples:", default="5", width=6)
        self.min_samples_entry.pack(fill=tk.X, padx=6, pady=3)

        self.scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Scale features", variable=self.scale_var).pack(anchor=tk.W, padx=6, pady=2)

        # Feature selection
        feat_frame = ttk.LabelFrame(left, text="Features (numeric only)")
        feat_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas = tk.Canvas(feat_frame, highlightthickness=0, height=150)
        vsb = ttk.Scrollbar(feat_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.feat_inner = ttk.Frame(canvas)
        self.feat_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.feat_inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.feat_vars: dict[str, tk.BooleanVar] = {}

        # Buttons
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(bf, text="Run Clustering", command=self._run).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Elbow Plot", command=self._elbow).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="PCA Scatter", command=self._pca_scatter).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)

        # ── Right ──
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        self.result_log = ScrolledLog(right, height=30)
        self.result_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _on_data_change(self):
        if self.dm.df is None:
            return
        for w in self.feat_inner.winfo_children():
            w.destroy()
        self.feat_vars.clear()
        for col in self.dm.df.select_dtypes("number").columns:
            var = tk.BooleanVar(value=True)
            self.feat_vars[col] = var
            ttk.Checkbutton(self.feat_inner, text=col, variable=var).pack(anchor=tk.W, padx=6, pady=1)

    def _get_X(self):
        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return None, None
        selected = [c for c, v in self.feat_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("PyWeka", "Select numeric features.")
            return None, None
        sub = df[selected].dropna()
        return sub.values, selected

    def _make_model(self):
        name = self.alg_combo.get()
        k = self.k_entry.get_int(3)
        eps = self.eps_entry.get_float(0.5)
        ms = self.min_samples_entry.get_int(5)
        if name == "K-Means":
            return name, KMeans(n_clusters=k, random_state=42, n_init=10)
        elif name == "Mini-Batch K-Means":
            return name, MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        elif name == "DBSCAN":
            return name, DBSCAN(eps=eps, min_samples=ms)
        elif name == "Agglomerative":
            return name, AgglomerativeClustering(n_clusters=k)
        elif name == "Mean Shift":
            return name, MeanShift()
        return name, KMeans(n_clusters=k, random_state=42, n_init=10)

    def _run(self):
        data = self._get_X()
        if data[0] is None:
            return
        X, feat_names = data
        name, model = self._make_model()
        self.status(f"Running {name}...")
        self.update_idletasks()

        try:
            res = run_clustering(X, name, model, scale=self.scale_var.get())
            self.results.append(res)
            self.result_log.append("=" * 70)
            self.result_log.append(res.summary_line())
            self.result_log.append(f"  Cluster sizes: {res.cluster_sizes}")
            self.result_log.append("=" * 70)
        except Exception as e:
            self.result_log.append(f"ERROR: {e}")
        self.status("Clustering complete.")

    def _elbow(self):
        data = self._get_X()
        if data[0] is None:
            return
        X, _ = data
        if self.scale_var.get():
            X = StandardScaler().fit_transform(X)

        K_range = range(2, 12)
        inertias, sils = [], []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sample = min(3000, len(X))
            idx = np.random.RandomState(42).choice(len(X), sample, replace=False)
            sils.append(silhouette_score(X[idx], labels[idx]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(K_range), inertias, "bo-")
        ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
        ax2.plot(list(K_range), sils, "rs-")
        ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.set_ylabel("Score")
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Elbow & Silhouette")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _pca_scatter(self):
        if not self.results:
            messagebox.showinfo("PyWeka", "Run clustering first.")
            return
        res = self.results[-1]
        X = res.X
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
        else:
            X2 = X

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X2[:, 0], X2[:, 1], c=res.labels, cmap="viridis",
                             alpha=0.4, s=8)
        ax.set_title(f"{res.name} – PCA Scatter (Sil={res.silhouette:.4f})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Cluster PCA Scatter")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear(self):
        self.result_log.clear()
        self.results.clear()
