"""
select_tab.py – Feature Selection panel (mirrors Weka ▸ Select Attributes).
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
from sklearn.preprocessing import LabelEncoder

from core.data_manager import DataManager
from core.evaluator import (
    select_features_kbest,
    select_features_importance,
    select_features_rfe,
)
from ui.widgets import ScrolledLog, LabeledCombo, LabeledEntry


class SelectTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self._last_ranked = []
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        # Target
        tf = ttk.LabelFrame(left, text="Target Variable")
        tf.pack(fill=tk.X, padx=4, pady=4)
        self.target_combo = LabeledCombo(tf, "Target:", [], width=24)
        self.target_combo.pack(fill=tk.X, padx=6, pady=4)

        # Task type
        tt = ttk.LabelFrame(left, text="Task Type")
        tt.pack(fill=tk.X, padx=4, pady=4)
        self.task_var = tk.StringVar(value="classification")
        ttk.Radiobutton(tt, text="Classification", variable=self.task_var,
                        value="classification").pack(anchor=tk.W, padx=6, pady=1)
        ttk.Radiobutton(tt, text="Regression", variable=self.task_var,
                        value="regression").pack(anchor=tk.W, padx=6, pady=1)

        # Method
        mf = ttk.LabelFrame(left, text="Selection Method")
        mf.pack(fill=tk.X, padx=4, pady=4)
        self.method_var = tk.StringVar(value="kbest")
        methods = [
            ("SelectKBest (univariate)", "kbest"),
            ("Random Forest Importance", "importance"),
            ("RFE (Recursive Feature Elimination)", "rfe"),
            ("Correlation Analysis", "correlation"),
        ]
        for text, val in methods:
            ttk.Radiobutton(mf, text=text, variable=self.method_var,
                            value=val).pack(anchor=tk.W, padx=6, pady=1)

        self.k_entry = LabeledEntry(mf, "Top K features:", default="10", width=5)
        self.k_entry.pack(fill=tk.X, padx=6, pady=4)

        # Buttons
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(bf, text="Run Selection", command=self._run).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Plot", command=self._plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)

        # ── Right ──
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        self.result_log = ScrolledLog(right, height=30)
        self.result_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _on_data_change(self):
        if self.dm.df is not None:
            cols = list(self.dm.df.columns)
            self.target_combo.set_values(cols, default=cols[-1] if cols else None)

    def _prepare_data(self):
        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return None, None, None
        target = self.target_combo.get()
        if not target:
            messagebox.showwarning("PyWeka", "Select a target.")
            return None, None, None

        num_cols = [c for c in df.select_dtypes("number").columns if c != target]
        if not num_cols:
            messagebox.showwarning("PyWeka", "No numeric features.")
            return None, None, None

        sub = df[num_cols + [target]].dropna()
        X = sub[num_cols].values
        if self.task_var.get() == "classification":
            le = LabelEncoder()
            y = le.fit_transform(sub[target].astype(str))
        else:
            if not pd.api.types.is_numeric_dtype(sub[target]):
                messagebox.showwarning("PyWeka", "Target must be numeric for regression.")
                return None, None, None
            y = sub[target].values
        return X, y, num_cols

    def _run(self):
        data = self._prepare_data()
        if data[0] is None:
            return
        X, y, feature_names = data
        method = self.method_var.get()
        k = self.k_entry.get_int(10)
        task = self.task_var.get()

        self.result_log.append("=" * 60)
        self.result_log.append(f"Feature Selection  |  Method: {method}  |  K: {k}  |  Task: {task}")
        self.result_log.append("=" * 60)
        self.status(f"Running feature selection ({method})...")
        self.update_idletasks()

        try:
            if method == "kbest":
                ranked = select_features_kbest(X, y, feature_names, k=k, task=task)
                header = "Feature Score"
                self._last_ranked = [(name, float(score)) for name, score in ranked]
            elif method == "importance":
                ranked = select_features_importance(X, y, feature_names, task=task)
                header = "Feature Importance"
                self._last_ranked = [(name, float(imp)) for name, imp in ranked]
            elif method == "rfe":
                ranked = select_features_rfe(X, y, feature_names, n_features=k, task=task)
                header = "Feature Rank (1=best)"
                self._last_ranked = [(name, int(rank)) for name, rank in ranked]
            elif method == "correlation":
                corr = pd.DataFrame(X, columns=feature_names).corrwith(pd.Series(y))
                ranked = sorted(corr.abs().items(), key=lambda x: x[1], reverse=True)
                header = "Feature |Correlation|"
                self._last_ranked = [(name, float(val)) for name, val in ranked]
            else:
                return

            self.result_log.append(f"\n{'Rank':<6}{header}")
            self.result_log.append("-" * 50)
            for i, (name, val) in enumerate(ranked[:k], 1):
                self.result_log.append(f"  {i:<4} {name:<30s} {val:.4f}")
            self.result_log.append("")

        except Exception as e:
            self.result_log.append(f"ERROR: {e}")
        self.status("Feature selection complete.")

    def _plot(self):
        if not self._last_ranked:
            messagebox.showinfo("PyWeka", "Run feature selection first.")
            return
        top = self._last_ranked[:20]
        names = [n for n, _ in top]
        values = [v for _, v in top]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names[::-1], values[::-1], color="steelblue")
        ax.set_title(f"Feature Selection – {self.method_var.get()}")
        ax.set_xlabel("Score / Importance")
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Feature Selection")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear(self):
        self.result_log.clear()
        self._last_ranked = []
