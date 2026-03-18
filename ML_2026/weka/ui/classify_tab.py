"""
classify_tab.py – Classification panel (mirrors Weka Explorer ▸ Classify).
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from core.data_manager import DataManager
from core.evaluator import (
    get_classifiers,
    run_classification,
    ClassificationResult,
)
from ui.widgets import ScrolledLog, LabeledCombo, LabeledEntry, CheckGroup


class ClassifyTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self.results: list[ClassificationResult] = []
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left panel: settings ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        # Target
        target_frame = ttk.LabelFrame(left, text="Target Variable")
        target_frame.pack(fill=tk.X, padx=4, pady=4)
        self.target_combo = LabeledCombo(target_frame, "Target:", [], width=24)
        self.target_combo.pack(fill=tk.X, padx=6, pady=4)

        # Classifier selection
        clf_frame = ttk.LabelFrame(left, text="Classifiers")
        clf_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        clf_names = list(get_classifiers().keys())
        # scrollable checkbutton list
        canvas = tk.Canvas(clf_frame, highlightthickness=0, height=200)
        vsb = ttk.Scrollbar(clf_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.clf_vars: dict[str, tk.BooleanVar] = {}
        for name in clf_names:
            var = tk.BooleanVar(value=True)
            self.clf_vars[name] = var
            ttk.Checkbutton(inner, text=name, variable=var).pack(anchor=tk.W, padx=6, pady=1)

        ttk.Button(left, text="Select All", command=self._select_all_clf).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(left, text="Deselect All", command=self._deselect_all_clf).pack(fill=tk.X, padx=4, pady=1)

        # Evaluation settings
        eval_frame = ttk.LabelFrame(left, text="Evaluation")
        eval_frame.pack(fill=tk.X, padx=4, pady=4)

        self.eval_mode = tk.StringVar(value="split")
        modes = [
            ("Percentage Split", "split"),
            ("Cross-Validation", "cv"),
            ("Use Training Set", "training_set"),
        ]
        for text, val in modes:
            ttk.Radiobutton(eval_frame, text=text, variable=self.eval_mode,
                            value=val).pack(anchor=tk.W, padx=6, pady=1)

        params = ttk.Frame(eval_frame)
        params.pack(fill=tk.X, padx=6, pady=4)
        self.test_size = LabeledEntry(params, "Test %:", default="20", width=5)
        self.test_size.pack(side=tk.LEFT, padx=(0, 12))
        self.cv_folds = LabeledEntry(params, "CV Folds:", default="10", width=5)
        self.cv_folds.pack(side=tk.LEFT)

        self.scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(eval_frame, text="Scale features (StandardScaler)",
                        variable=self.scale_var).pack(anchor=tk.W, padx=6, pady=2)

        # Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(btn_frame, text="Start", command=self._run).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Confusion Matrix", command=self._show_cm).pack(side=tk.LEFT, padx=4)

        # ── Right panel: results ──
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        self.result_log = ScrolledLog(right, height=30)
        self.result_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Helpers ───────────────────────────────────────────────────────
    def _on_data_change(self):
        if self.dm.df is not None:
            cols = list(self.dm.df.columns)
            self.target_combo.set_values(cols, default=cols[-1] if cols else None)

    def _select_all_clf(self):
        for v in self.clf_vars.values():
            v.set(True)

    def _deselect_all_clf(self):
        for v in self.clf_vars.values():
            v.set(False)

    def _prepare_data(self):
        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return None, None, None
        target = self.target_combo.get()
        if not target or target not in df.columns:
            messagebox.showwarning("PyWeka", "Select a valid target variable.")
            return None, None, None

        # Features: only numeric columns (excluding target)
        num_cols = [c for c in df.select_dtypes("number").columns if c != target]
        if not num_cols:
            messagebox.showwarning("PyWeka", "No numeric feature columns found.\n"
                                   "Encode categorical columns in Preprocess tab first.")
            return None, None, None

        sub = df[num_cols + [target]].dropna()
        if len(sub) < 10:
            messagebox.showwarning("PyWeka", "Not enough valid rows after dropping NaNs.")
            return None, None, None

        X = sub[num_cols].values
        le = LabelEncoder()
        y = le.fit_transform(sub[target].astype(str))
        classes = list(le.classes_)
        return X, y, classes

    # ── Run ───────────────────────────────────────────────────────────
    def _run(self):
        data = self._prepare_data()
        if data[0] is None:
            return
        X, y, classes = data
        selected = [n for n, v in self.clf_vars.items() if v.get()]
        if not selected:
            messagebox.showinfo("PyWeka", "Select at least one classifier.")
            return

        mode = self.eval_mode.get()
        test_pct = self.test_size.get_float(20) / 100
        folds = self.cv_folds.get_int(10)
        scale = self.scale_var.get()

        self.result_log.append("=" * 70)
        self.result_log.append(f"Classification  |  Mode: {mode}  |  "
                               f"Test: {test_pct*100:.0f}%  |  Folds: {folds}  |  Scale: {scale}")
        self.result_log.append(f"Instances: {len(y)}  |  Features: {X.shape[1]}  |  "
                               f"Classes: {len(classes)} {classes}")
        self.result_log.append("=" * 70)
        self.status("Running classifiers...")
        self.update_idletasks()

        all_classifiers = get_classifiers()
        self.results.clear()

        for name in selected:
            if name not in all_classifiers:
                continue
            model = all_classifiers[name]
            try:
                res = run_classification(
                    X, y, name, model,
                    eval_mode=mode, test_size=test_pct,
                    cv_folds=folds, scale=scale, classes=classes,
                )
                self.results.append(res)
                self.result_log.append(res.summary_line())
            except Exception as e:
                self.result_log.append(f"  {name}: ERROR - {e}")
            self.update_idletasks()

        if self.results:
            best = max(self.results, key=lambda r: r.f1)
            self.result_log.append("-" * 70)
            self.result_log.append(f"Best: {best.name}  F1={best.f1:.4f}")
            self.result_log.append("")

            # Detailed report for best
            self.result_log.append(f"--- Classification Report ({best.name}) ---")
            self.result_log.append(best.report)

        self.status("Classification complete.")

    def _clear(self):
        self.result_log.clear()
        self.results.clear()

    def _show_cm(self):
        if not self.results:
            messagebox.showinfo("PyWeka", "Run classification first.")
            return
        best = max(self.results, key=lambda r: r.f1)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(best.cm, display_labels=best.classes)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix – {best.name} (F1={best.f1:.4f})")
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title(f"Confusion Matrix – {best.name}")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
