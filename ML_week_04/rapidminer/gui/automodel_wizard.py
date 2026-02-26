"""
automodel_wizard.py – A 5‑step guided wizard that:
  1. Load data (CSV / Excel / Parquet / JSON)
  2. Select task type (Classification / Regression / Clustering)
  3. Choose target column
  4. Select features + pre‑processing
  5. Train & compare models → leaderboard
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from gui.theme import C, F


# ═══════════════════════════════════════════════════════════════════════════
# AutoModelWizard
# ═══════════════════════════════════════════════════════════════════════════

class AutoModelWizard(tk.Toplevel):
    """Modal 5‑step wizard for automatic model selection."""

    MODELS_CLF = {
        "Logistic Regression": ("sklearn.linear_model", "LogisticRegression",
                                {"max_iter": 1000}),
        "Decision Tree":       ("sklearn.tree", "DecisionTreeClassifier", {}),
        "Random Forest":       ("sklearn.ensemble", "RandomForestClassifier",
                                {"n_estimators": 100}),
        "Gradient Boosting":   ("sklearn.ensemble", "GradientBoostingClassifier",
                                {"n_estimators": 100}),
        "SVM":                 ("sklearn.svm", "SVC", {"probability": True}),
        "KNN":                 ("sklearn.neighbors", "KNeighborsClassifier", {}),
        "Naive Bayes":         ("sklearn.naive_bayes", "GaussianNB", {}),
    }
    MODELS_REG = {
        "Linear Regression": ("sklearn.linear_model", "LinearRegression", {}),
        "Ridge":             ("sklearn.linear_model", "Ridge", {}),
        "Lasso":             ("sklearn.linear_model", "Lasso", {}),
        "Decision Tree":     ("sklearn.tree", "DecisionTreeRegressor", {}),
        "Random Forest":     ("sklearn.ensemble", "RandomForestRegressor",
                              {"n_estimators": 100}),
        "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingRegressor",
                              {"n_estimators": 100}),
    }
    MODELS_CLU = {
        "K‑Means":         ("sklearn.cluster", "KMeans", {"n_clusters": 3}),
        "DBSCAN":          ("sklearn.cluster", "DBSCAN", {}),
        "Agglomerative":   ("sklearn.cluster", "AgglomerativeClustering",
                            {"n_clusters": 3}),
    }

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.title("AutoModel Wizard")
        self.configure(bg=C.BG)
        self.geometry("720x540")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.df: Optional[pd.DataFrame] = None
        self.task: str = "classification"
        self.target: str = ""
        self.features: List[str] = []
        self.results: List[Dict[str, Any]] = []

        self.on_complete: Optional[Callable] = None

        self._step = 0
        self._frames: List[ttk.Frame] = []

        # Title bar
        ttk.Label(self, text="✨ AutoModel Wizard", style="Title.TLabel"
                  ).pack(fill="x", padx=16, pady=(12, 4))

        # Container
        self._container = ttk.Frame(self, style="Panel.TFrame")
        self._container.pack(fill="both", expand=True, padx=16, pady=8)

        # Bottom nav
        nav = ttk.Frame(self, style="TFrame")
        nav.pack(fill="x", padx=16, pady=8)
        self._btn_back = ttk.Button(nav, text="◀ Back", command=self._back,
                                    style="TButton")
        self._btn_back.pack(side="left")
        self._btn_next = ttk.Button(nav, text="Next ▶", command=self._next,
                                    style="Accent.TButton")
        self._btn_next.pack(side="right")

        self._build_steps()
        self._show_step(0)

    # ── Build each step ─────────────────────────────────────────────────

    def _build_steps(self) -> None:
        # Step 0 – Load data
        f0 = ttk.Frame(self._container, style="Panel.TFrame")
        ttk.Label(f0, text="Step 1: Load your dataset",
                  style="Heading.TLabel").pack(anchor="w", pady=(8, 4))
        ttk.Label(f0, text="Supported formats: CSV, Excel, JSON, Parquet",
                  style="Dim.TLabel").pack(anchor="w")
        self._file_var = tk.StringVar()
        ef = ttk.Frame(f0, style="Panel.TFrame")
        ef.pack(fill="x", pady=8)
        ttk.Entry(ef, textvariable=self._file_var, style="TEntry"
                  ).pack(side="left", fill="x", expand=True)
        ttk.Button(ef, text="Browse…", command=self._browse,
                   style="TButton").pack(side="right", padx=4)
        self._load_status = ttk.Label(f0, text="", style="Panel.TLabel")
        self._load_status.pack(anchor="w")
        self._frames.append(f0)

        # Step 1 – Task type
        f1 = ttk.Frame(self._container, style="Panel.TFrame")
        ttk.Label(f1, text="Step 2: Select task type",
                  style="Heading.TLabel").pack(anchor="w", pady=(8, 4))
        self._task_var = tk.StringVar(value="classification")
        for txt, val in [("Classification", "classification"),
                         ("Regression", "regression"),
                         ("Clustering", "clustering")]:
            ttk.Radiobutton(f1, text=txt, value=val, variable=self._task_var,
                            style="TRadiobutton").pack(anchor="w", padx=16,
                                                       pady=2)
        self._frames.append(f1)

        # Step 2 – Target column
        f2 = ttk.Frame(self._container, style="Panel.TFrame")
        ttk.Label(f2, text="Step 3: Select target column",
                  style="Heading.TLabel").pack(anchor="w", pady=(8, 4))
        self._target_combo = ttk.Combobox(f2, state="readonly",
                                          style="TCombobox")
        self._target_combo.pack(fill="x", padx=8, pady=4)
        self._target_info = ttk.Label(f2, text="", style="Dim.TLabel",
                                      wraplength=600)
        self._target_info.pack(anchor="w", padx=8)
        self._frames.append(f2)

        # Step 3 – Features
        f3 = ttk.Frame(self._container, style="Panel.TFrame")
        ttk.Label(f3, text="Step 4: Select features",
                  style="Heading.TLabel").pack(anchor="w", pady=(8, 4))
        self._feat_frame = ttk.Frame(f3, style="Panel.TFrame")
        self._feat_frame.pack(fill="both", expand=True, padx=8)
        self._feat_vars: Dict[str, tk.BooleanVar] = {}
        self._frames.append(f3)

        # Step 4 – Train & leaderboard
        f4 = ttk.Frame(self._container, style="Panel.TFrame")
        ttk.Label(f4, text="Step 5: Training & Results",
                  style="Heading.TLabel").pack(anchor="w", pady=(8, 4))
        self._progress = ttk.Progressbar(f4, mode="determinate",
                                         style="TProgressbar")
        self._progress.pack(fill="x", padx=8, pady=8)
        self._prog_label = ttk.Label(f4, text="", style="Panel.TLabel")
        self._prog_label.pack(anchor="w", padx=8)

        # Leaderboard tree
        cols = ("Model", "Score", "Std")
        self._leader = ttk.Treeview(f4, columns=cols, show="headings",
                                    height=8, style="Panel.Treeview")
        for c in cols:
            self._leader.heading(c, text=c)
            self._leader.column(c, width=180)
        self._leader.pack(fill="both", expand=True, padx=8, pady=4)
        self._frames.append(f4)

    # ── Navigation ──────────────────────────────────────────────────────

    def _show_step(self, idx: int) -> None:
        for f in self._frames:
            f.pack_forget()
        self._frames[idx].pack(fill="both", expand=True)
        self._step = idx
        self._btn_back.configure(state="normal" if idx > 0 else "disabled")
        self._btn_next.configure(
            text="🚀 Train!" if idx == len(self._frames) - 1 else "Next ▶")

    def _next(self) -> None:
        if self._step == 0:
            if not self._load_data():
                return
        elif self._step == 1:
            self.task = self._task_var.get()
            self._populate_target()
        elif self._step == 2:
            self.target = self._target_combo.get()
            if self.task != "clustering" and not self.target:
                return
            self._populate_features()
        elif self._step == 3:
            self.features = [c for c, v in self._feat_vars.items() if v.get()]
            if not self.features:
                return
            self._train()
            return  # _train handles step transition
        elif self._step == 4:
            self.destroy()
            return

        if self._step < len(self._frames) - 1:
            self._show_step(self._step + 1)

    def _back(self) -> None:
        if self._step > 0:
            self._show_step(self._step - 1)

    # ── Step implementations ────────────────────────────────────────────

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.csv *.xlsx *.json *.parquet"),
                       ("All files", "*.*")])
        if path:
            self._file_var.set(path)

    def _load_data(self) -> bool:
        path = Path(self._file_var.get())
        if not path.exists():
            self._load_status.configure(text="⚠ File not found.",
                                        foreground=C.ERROR)
            return False
        try:
            ext = path.suffix.lower()
            if ext == ".csv":
                self.df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"):
                self.df = pd.read_excel(path)
            elif ext == ".json":
                self.df = pd.read_json(path)
            elif ext == ".parquet":
                self.df = pd.read_parquet(path)
            else:
                self._load_status.configure(text="⚠ Unsupported format.",
                                            foreground=C.ERROR)
                return False
            self._load_status.configure(
                text=f"✓ Loaded {len(self.df):,} rows × {len(self.df.columns)} cols",
                foreground=C.SUCCESS)
            return True
        except Exception as exc:
            self._load_status.configure(text=f"⚠ {exc}", foreground=C.ERROR)
            return False

    def _populate_target(self) -> None:
        if self.df is None:
            return
        cols = list(self.df.columns)
        self._target_combo["values"] = cols
        if cols:
            self._target_combo.set(cols[-1])
        if self.task == "clustering":
            self._target_info.configure(
                text="For clustering, target is optional (used only for evaluation).")

    def _populate_features(self) -> None:
        for w in self._feat_frame.winfo_children():
            w.destroy()
        self._feat_vars.clear()
        if self.df is None:
            return
        cols = [c for c in self.df.columns if c != self.target]
        # Select all numeric by default
        for c in cols:
            var = tk.BooleanVar(value=self.df[c].dtype.kind in "iufb")
            self._feat_vars[c] = var
            ttk.Checkbutton(self._feat_frame, text=c, variable=var,
                            style="TCheckbutton").pack(anchor="w")

    def _train(self) -> None:
        self._show_step(4)
        self._leader.delete(*self._leader.get_children())
        self._progress["value"] = 0
        self._prog_label.configure(text="Preparing…")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        from importlib import import_module

        df = self.df[self.features + ([self.target] if self.target else [])].copy()
        df = df.dropna()

        # Encode categoricals
        for c in df.select_dtypes(include=["object", "category"]).columns:
            df[c] = df[c].astype("category").cat.codes

        X = df[self.features].values
        y = df[self.target].values if self.target else None

        models = (self.MODELS_CLF if self.task == "classification"
                  else self.MODELS_REG if self.task == "regression"
                  else self.MODELS_CLU)

        scoring = ("accuracy" if self.task == "classification"
                   else "r2" if self.task == "regression" else None)

        total = len(models)
        results: List[Dict[str, Any]] = []

        for i, (name, (mod_path, cls_name, params)) in enumerate(models.items()):
            self.after(0, lambda n=name: self._prog_label.configure(
                text=f"Training {n}…"))
            self.after(0, lambda v=(i / total * 100): self._progress.configure(
                value=v))

            try:
                mod = import_module(mod_path)
                cls = getattr(mod, cls_name)
                est = cls(**params)

                if self.task in ("classification", "regression"):
                    scores = cross_val_score(est, X, y, cv=5,
                                             scoring=scoring)
                    mean_score = scores.mean()
                    std_score = scores.std()
                else:  # clustering
                    from sklearn.metrics import silhouette_score
                    est.fit(X)
                    labels = est.labels_ if hasattr(est, "labels_") else est.predict(X)
                    n_unique = len(set(labels))
                    if n_unique > 1:
                        mean_score = silhouette_score(X, labels)
                    else:
                        mean_score = -1.0
                    std_score = 0.0

                results.append({
                    "name": name, "score": mean_score, "std": std_score,
                    "estimator": est,
                })
            except Exception as exc:
                results.append({
                    "name": name, "score": float("nan"), "std": 0.0,
                    "error": str(exc),
                })

        # Sort leaderboard
        results.sort(key=lambda r: r["score"] if not np.isnan(r["score"])
                     else -999, reverse=True)
        self.results = results

        # Update UI on main thread
        self.after(0, lambda: self._show_leaderboard(results))

    def _show_leaderboard(self, results: List[Dict]) -> None:
        self._progress["value"] = 100
        self._prog_label.configure(text="✓ Training complete!")
        for r in results:
            score_str = f"{r['score']:.4f}" if not np.isnan(r["score"]) else "Error"
            std_str = f"±{r['std']:.4f}" if not np.isnan(r["score"]) else ""
            tag = "best" if r is results[0] else ""
            self._leader.insert("", "end",
                                values=(r["name"], score_str, std_str),
                                tags=(tag,))
        self._leader.tag_configure("best", foreground=C.SUCCESS)
