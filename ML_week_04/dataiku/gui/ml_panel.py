"""
ML Panel – GUI for the Machine Learning Lab.

Provides target/feature selectors, algorithm picker with hyperparameter
controls, train button, results display, confusion matrix, and
feature-importance chart.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.ml import (
    ALL_ALGORITHMS,
    CLASSIFICATION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
    MLLab,
    REGRESSION_ALGORITHMS,
    TrainResult,
)
from gui.themes import Colors, FONT_BOLD, FONT_CODE, FONT_NORMAL, FONT_SMALL
from utils.helpers import run_in_background


# ---------------------------------------------------------------------------
# MLPanel
# ---------------------------------------------------------------------------

class MLPanel(ttk.Frame):
    """Machine Learning Lab panel with full train/evaluate workflow."""

    def __init__(
        self,
        parent: tk.Widget,
        get_dataset_names: Callable[[], List[str]],
        get_dataframe: Callable[[str], pd.DataFrame],
        set_status: Callable[[str], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._get_names = get_dataset_names
        self._get_df = get_dataframe
        self._set_status = set_status
        self._lab = MLLab()
        self._param_widgets: Dict[str, tk.Widget] = {}
        self._param_vars: Dict[str, tk.StringVar] = {}
        self._build_ui()

    # -- UI build ------------------------------------------------------------

    def _build_ui(self) -> None:
        # Main horizontal paned window: left config | right results
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # --- Left: configuration ---
        left = ttk.Frame(pw)
        pw.add(left, weight=1)

        # Dataset selector
        ds_frame = ttk.LabelFrame(left, text="Dataset")
        ds_frame.pack(fill=tk.X, padx=6, pady=4)

        row = ttk.Frame(ds_frame)
        row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(row, text="Dataset:").pack(side=tk.LEFT)
        self._ds_var = tk.StringVar()
        self._ds_combo = ttk.Combobox(row, textvariable=self._ds_var, state="readonly", width=22)
        self._ds_combo.pack(side=tk.LEFT, padx=4)
        self._ds_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)
        ttk.Button(row, text="\u21bb", width=3, command=self._refresh_datasets).pack(side=tk.LEFT)

        # Target selector
        target_frame = ttk.LabelFrame(left, text="Target Column")
        target_frame.pack(fill=tk.X, padx=6, pady=4)
        row2 = ttk.Frame(target_frame)
        row2.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(row2, text="Target:").pack(side=tk.LEFT)
        self._target_var = tk.StringVar()
        self._target_combo = ttk.Combobox(row2, textvariable=self._target_var, state="readonly", width=22)
        self._target_combo.pack(side=tk.LEFT, padx=4)

        # Feature selector (multi-select listbox)
        feat_frame = ttk.LabelFrame(left, text="Feature Columns")
        feat_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._feat_listbox = tk.Listbox(
            feat_frame, selectmode=tk.MULTIPLE, height=8,
            bg=Colors.INPUT_BG, fg=Colors.TEXT, selectbackground=Colors.ACCENT,
            font=FONT_NORMAL, relief=tk.FLAT,
        )
        self._feat_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        btn_row = ttk.Frame(feat_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(btn_row, text="Select All", command=self._select_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_features).pack(side=tk.LEFT, padx=2)

        # Task + algorithm
        algo_frame = ttk.LabelFrame(left, text="Algorithm")
        algo_frame.pack(fill=tk.X, padx=6, pady=4)

        r1 = ttk.Frame(algo_frame)
        r1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r1, text="Task:").pack(side=tk.LEFT)
        self._task_var = tk.StringVar(value="classification")
        self._task_combo = ttk.Combobox(
            r1, textvariable=self._task_var, state="readonly", width=18,
            values=["classification", "regression", "clustering"],
        )
        self._task_combo.pack(side=tk.LEFT, padx=4)
        self._task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)

        r2 = ttk.Frame(algo_frame)
        r2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r2, text="Algorithm:").pack(side=tk.LEFT)
        self._algo_var = tk.StringVar()
        self._algo_combo = ttk.Combobox(r2, textvariable=self._algo_var, state="readonly", width=22)
        self._algo_combo.pack(side=tk.LEFT, padx=4)
        self._algo_combo.bind("<<ComboboxSelected>>", self._on_algo_changed)

        # Hyperparameters
        self._hp_frame = ttk.LabelFrame(left, text="Hyperparameters")
        self._hp_frame.pack(fill=tk.X, padx=6, pady=4)

        # Train/test split
        split_frame = ttk.LabelFrame(left, text="Train / Test Split")
        split_frame.pack(fill=tk.X, padx=6, pady=4)
        split_row = ttk.Frame(split_frame)
        split_row.pack(fill=tk.X, padx=4, pady=4)
        self._split_var = tk.DoubleVar(value=0.2)
        ttk.Label(split_row, text="Test size:").pack(side=tk.LEFT)
        self._split_scale = ttk.Scale(
            split_row, from_=0.05, to=0.5, variable=self._split_var,
            orient=tk.HORIZONTAL, command=self._on_split_change,
        )
        self._split_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._split_label = ttk.Label(split_row, text="20%", width=6)
        self._split_label.pack(side=tk.LEFT)

        # Train button
        train_row = ttk.Frame(left)
        train_row.pack(fill=tk.X, padx=6, pady=8)
        ttk.Button(train_row, text="\u25b6  Train Model", style="Accent.TButton",
                    command=self._train).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_row, text="Export .pkl", command=self._export_model).pack(side=tk.LEFT, padx=4)
        self._progress = ttk.Progressbar(train_row, mode="indeterminate", length=120)
        self._progress.pack(side=tk.LEFT, padx=8)

        # --- Right: results ---
        right = ttk.Frame(pw)
        pw.add(right, weight=2)

        self._results_nb = ttk.Notebook(right)
        self._results_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Metrics tab
        self._metrics_frame = ttk.Frame(self._results_nb)
        self._results_nb.add(self._metrics_frame, text="Metrics")
        self._metrics_text = tk.Text(
            self._metrics_frame, wrap=tk.WORD, height=16,
            bg=Colors.PANEL, fg=Colors.TEXT, font=FONT_CODE,
            relief=tk.FLAT, insertbackground=Colors.TEXT,
        )
        self._metrics_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Confusion matrix / chart tab
        self._chart_frame = ttk.Frame(self._results_nb)
        self._results_nb.add(self._chart_frame, text="Charts")

        # Feature importance tab
        self._fi_frame = ttk.Frame(self._results_nb)
        self._results_nb.add(self._fi_frame, text="Feature Importance")

        # Initial population
        self._on_task_changed(None)

    # -- callbacks -----------------------------------------------------------

    def _refresh_datasets(self) -> None:
        names = self._get_names()
        self._ds_combo["values"] = names
        if names and not self._ds_var.get():
            self._ds_var.set(names[0])
            self._on_dataset_changed(None)

    def _on_dataset_changed(self, _event: Any) -> None:
        name = self._ds_var.get()
        if not name:
            return
        try:
            df = self._get_df(name)
        except Exception:
            return
        cols = list(df.columns)
        self._target_combo["values"] = cols
        self._feat_listbox.delete(0, tk.END)
        for c in cols:
            self._feat_listbox.insert(tk.END, c)
        if cols:
            self._target_var.set(cols[-1])

    def _on_task_changed(self, _event: Any) -> None:
        task = self._task_var.get()
        registry = ALL_ALGORITHMS.get(task, {})
        algo_names = list(registry.keys())
        self._algo_combo["values"] = algo_names
        if algo_names:
            self._algo_var.set(algo_names[0])
            self._on_algo_changed(None)

    def _on_algo_changed(self, _event: Any) -> None:
        task = self._task_var.get()
        algo_name = self._algo_var.get()
        registry = ALL_ALGORITHMS.get(task, {})
        algo_info = registry.get(algo_name, {})
        params = algo_info.get("params", {})

        # Rebuild hyperparameter widgets
        for w in self._hp_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()
        self._param_widgets.clear()

        for pname, pspec in params.items():
            row = ttk.Frame(self._hp_frame)
            row.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(row, text=f"{pname}:").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(pspec.get("default", "")))
            if pspec["type"] == "choice":
                w = ttk.Combobox(row, textvariable=var, values=pspec.get("choices", []),
                                 state="readonly", width=14)
            else:
                w = ttk.Entry(row, textvariable=var, width=14)
            w.pack(side=tk.LEFT, padx=4)
            self._param_vars[pname] = var
            self._param_widgets[pname] = w

    def _on_split_change(self, val: str) -> None:
        pct = int(float(val) * 100)
        self._split_label.config(text=f"{pct}%")

    def _select_all_features(self) -> None:
        self._feat_listbox.select_set(0, tk.END)

    def _deselect_features(self) -> None:
        self._feat_listbox.select_clear(0, tk.END)

    # -- training ------------------------------------------------------------

    def _train(self) -> None:
        name = self._ds_var.get()
        if not name:
            messagebox.showwarning("ML Lab", "Select a dataset first.")
            return
        try:
            df = self._get_df(name)
        except Exception as e:
            messagebox.showerror("ML Lab", str(e))
            return

        target = self._target_var.get()
        selected_indices = self._feat_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("ML Lab", "Select at least one feature column.")
            return
        features = [self._feat_listbox.get(i) for i in selected_indices]
        # Remove target from features if present
        features = [f for f in features if f != target]

        task = self._task_var.get()
        algo = self._algo_var.get()
        params = {k: v.get() for k, v in self._param_vars.items()}
        test_size = self._split_var.get()

        self._progress.start()
        self._set_status("Training model…")

        def _do_train() -> TrainResult:
            return self._lab.train(
                df=df,
                target_col=target,
                feature_cols=features,
                task=task,
                algorithm_name=algo,
                params=params,
                test_size=test_size,
            )

        def _on_done(result: TrainResult) -> None:
            self._progress.stop()
            self._set_status(f"Training complete – {algo}")
            self._show_results(result)

        def _on_error(exc: Exception) -> None:
            self._progress.stop()
            self._set_status("Training failed")
            messagebox.showerror("ML Lab", f"Training failed:\n{exc}")

        run_in_background(_do_train, on_done=_on_done, on_error=_on_error)

    # -- display results -----------------------------------------------------

    def _show_results(self, result: TrainResult) -> None:
        """Populate the results tabs with metrics, charts, and feature importances."""
        # Metrics
        self._metrics_text.delete("1.0", tk.END)
        lines = [
            f"Task:       {result.task}",
            f"Algorithm:  {result.algorithm}",
            f"Target:     {result.target_name}",
            f"Features:   {', '.join(result.feature_names)}",
            "",
            "─── Metrics ───",
        ]
        for k, v in result.metrics.items():
            lines.append(f"  {k:18s}: {v:.4f}")
        self._metrics_text.insert("1.0", "\n".join(lines))

        # Charts: confusion matrix for classification, residuals for regression
        for child in self._chart_frame.winfo_children():
            child.destroy()

        plt.rcParams.update({
            "figure.facecolor": Colors.PANEL,
            "axes.facecolor": Colors.CANVAS_BG,
            "axes.edgecolor": Colors.BORDER,
            "axes.labelcolor": Colors.TEXT,
            "text.color": Colors.TEXT,
            "xtick.color": Colors.TEXT_DIM,
            "ytick.color": Colors.TEXT_DIM,
        })

        if result.task == "classification" and result.confusion_mat is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            import seaborn as sns
            sns.heatmap(result.confusion_mat, annot=True, fmt="d",
                        cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            self._embed_fig(fig, self._chart_frame)

        elif result.task == "regression" and result.y_test is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(result.y_test, result.y_pred, alpha=0.5, color=Colors.ACCENT, s=12)
            mn = min(result.y_test.min(), result.y_pred.min())
            mx = max(result.y_test.max(), result.y_pred.max())
            ax.plot([mn, mx], [mn, mx], "--", color=Colors.ERROR, linewidth=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            fig.tight_layout()
            self._embed_fig(fig, self._chart_frame)

        # Feature importance
        for child in self._fi_frame.winfo_children():
            child.destroy()
        if result.feature_importances is not None:
            fi = result.feature_importances
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sorted_idx = np.argsort(fi)
            ax2.barh(
                [result.feature_names[i] for i in sorted_idx],
                fi[sorted_idx],
                color=Colors.ACCENT,
            )
            ax2.set_title("Feature Importance")
            fig2.tight_layout()
            self._embed_fig(fig2, self._fi_frame)

    # -- export --------------------------------------------------------------

    def _export_model(self) -> None:
        if not self._lab.results:
            messagebox.showinfo("Export", "No trained model to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if path:
            from pathlib import Path
            MLLab.export_model(self._lab.results[-1].model, Path(path))
            messagebox.showinfo("Exported", f"Model saved to {path}")
            self._set_status(f"Model exported → {path}")

    # -- embed figure --------------------------------------------------------

    def _embed_fig(self, fig: Figure, parent: ttk.Frame) -> None:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
