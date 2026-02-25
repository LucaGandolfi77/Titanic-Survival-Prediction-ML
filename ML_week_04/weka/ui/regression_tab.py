"""
regression_tab.py – Regression panel (mirrors Weka Explorer concepts).
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

from core.data_manager import DataManager
from core.evaluator import get_regressors, run_regression, RegressionResult
from ui.widgets import ScrolledLog, LabeledCombo, LabeledEntry


class RegressionTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self.results: list[RegressionResult] = []
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        # Target
        tf = ttk.LabelFrame(left, text="Target Variable (numeric)")
        tf.pack(fill=tk.X, padx=4, pady=4)
        self.target_combo = LabeledCombo(tf, "Target:", [], width=24)
        self.target_combo.pack(fill=tk.X, padx=6, pady=4)

        # Regressor selection
        reg_frame = ttk.LabelFrame(left, text="Regressors")
        reg_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas = tk.Canvas(reg_frame, highlightthickness=0, height=180)
        vsb = ttk.Scrollbar(reg_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        reg_names = list(get_regressors().keys())
        self.reg_vars: dict[str, tk.BooleanVar] = {}
        for name in reg_names:
            var = tk.BooleanVar(value=True)
            self.reg_vars[name] = var
            ttk.Checkbutton(inner, text=name, variable=var).pack(anchor=tk.W, padx=6, pady=1)

        ttk.Button(left, text="Select All", command=lambda: [v.set(True) for v in self.reg_vars.values()]).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(left, text="Deselect All", command=lambda: [v.set(False) for v in self.reg_vars.values()]).pack(fill=tk.X, padx=4, pady=1)

        # Eval settings
        ef = ttk.LabelFrame(left, text="Evaluation")
        ef.pack(fill=tk.X, padx=4, pady=4)
        self.eval_mode = tk.StringVar(value="split")
        for text, val in [("Percentage Split", "split"), ("Cross-Validation", "cv"),
                          ("Use Training Set", "training_set")]:
            ttk.Radiobutton(ef, text=text, variable=self.eval_mode, value=val).pack(anchor=tk.W, padx=6, pady=1)

        pf = ttk.Frame(ef)
        pf.pack(fill=tk.X, padx=6, pady=4)
        self.test_size = LabeledEntry(pf, "Test %:", default="20", width=5)
        self.test_size.pack(side=tk.LEFT, padx=(0, 12))
        self.cv_folds = LabeledEntry(pf, "CV Folds:", default="10", width=5)
        self.cv_folds.pack(side=tk.LEFT)

        self.scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ef, text="Scale features", variable=self.scale_var).pack(anchor=tk.W, padx=6, pady=2)

        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(bf, text="Start", command=self._run).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Actual vs Predicted", command=self._show_plot).pack(side=tk.LEFT, padx=4)

        # ── Right ──
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        self.result_log = ScrolledLog(right, height=30)
        self.result_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _on_data_change(self):
        if self.dm.df is not None:
            num_cols = list(self.dm.df.select_dtypes("number").columns)
            self.target_combo.set_values(num_cols, default=num_cols[-1] if num_cols else None)

    def _prepare_data(self):
        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return None, None
        target = self.target_combo.get()
        if not target or target not in df.columns:
            messagebox.showwarning("PyWeka", "Select a valid numeric target.")
            return None, None
        if not pd.api.types.is_numeric_dtype(df[target]):
            messagebox.showwarning("PyWeka", f"'{target}' is not numeric.")
            return None, None

        num_cols = [c for c in df.select_dtypes("number").columns if c != target]
        if not num_cols:
            messagebox.showwarning("PyWeka", "No numeric features found.")
            return None, None

        sub = df[num_cols + [target]].dropna()
        X = sub[num_cols].values
        y = sub[target].values
        return X, y

    def _run(self):
        data = self._prepare_data()
        if data[0] is None:
            return
        X, y = data
        selected = [n for n, v in self.reg_vars.items() if v.get()]
        if not selected:
            messagebox.showinfo("PyWeka", "Select at least one regressor.")
            return

        mode = self.eval_mode.get()
        test_pct = self.test_size.get_float(20) / 100
        folds = self.cv_folds.get_int(10)
        scale = self.scale_var.get()

        self.result_log.append("=" * 70)
        self.result_log.append(f"Regression  |  Mode: {mode}  |  Test: {test_pct*100:.0f}%  |  Folds: {folds}")
        self.result_log.append(f"Instances: {len(y)}  |  Features: {X.shape[1]}")
        self.result_log.append("=" * 70)
        self.status("Running regressors...")
        self.update_idletasks()

        all_regs = get_regressors()
        self.results.clear()

        for name in selected:
            if name not in all_regs:
                continue
            model = all_regs[name]
            try:
                res = run_regression(
                    X, y, name, model,
                    eval_mode=mode, test_size=test_pct,
                    cv_folds=folds, scale=scale,
                )
                self.results.append(res)
                self.result_log.append(res.summary_line())
            except Exception as e:
                self.result_log.append(f"  {name}: ERROR - {e}")
            self.update_idletasks()

        if self.results:
            best = max(self.results, key=lambda r: r.r2)
            self.result_log.append("-" * 70)
            self.result_log.append(f"Best: {best.name}  R2={best.r2:.4f}  RMSE={best.rmse:.4f}")
            self.result_log.append("")
        self.status("Regression complete.")

    def _clear(self):
        self.result_log.clear()
        self.results.clear()

    def _show_plot(self):
        if not self.results:
            messagebox.showinfo("PyWeka", "Run regression first.")
            return
        n = len(self.results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for idx, res in enumerate(self.results):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.scatter(res.y_true, res.y_pred, alpha=0.15, s=6, c="teal")
            lims = [min(res.y_true.min(), res.y_pred.min()),
                    max(res.y_true.max(), res.y_pred.max())]
            ax.plot(lims, lims, "r--", lw=1.5)
            ax.set_title(f"{res.name}\nR2={res.r2:.4f}", fontsize=9)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Regression – Actual vs Predicted")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
