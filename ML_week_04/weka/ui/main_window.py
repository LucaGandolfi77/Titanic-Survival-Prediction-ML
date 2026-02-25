"""
main_window.py – Main application window with tabbed interface
                  (mirrors Weka Explorer layout).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.data_manager import DataManager
from ui.preprocess_tab import PreprocessTab
from ui.classify_tab import ClassifyTab
from ui.regression_tab import RegressionTab
from ui.cluster_tab import ClusterTab
from ui.select_tab import SelectTab
from ui.visualize_tab import VisualizeTab
from ui.associate_tab import AssociateTab
from ui.widgets import StatusBar


class MainWindow:
    """Root application window."""

    TITLE = "PyWeka – Machine Learning Explorer"
    MIN_W, MIN_H = 1100, 750

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(self.TITLE)
        self.root.geometry("1280x850")
        self.root.minsize(self.MIN_W, self.MIN_H)

        # Apply a nice theme
        style = ttk.Style(self.root)
        available_themes = style.theme_names()
        for preferred in ("clam", "aqua", "alt", "default"):
            if preferred in available_themes:
                style.theme_use(preferred)
                break

        # Custom styles
        style.configure("TNotebook.Tab", padding=[14, 6], font=("Helvetica", 11))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10), padding=4)
        style.configure("Treeview", font=("Courier", 10), rowheight=22)
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

        self.dm = DataManager()

        self._build_menu()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

    # ── Menu bar ──────────────────────────────────────────────────────
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Dataset...", command=self._open_file, accelerator="Cmd+O")
        file_menu.add_command(label="Save Dataset...", command=self._save_file, accelerator="Cmd+S")
        file_menu.add_separator()
        file_menu.add_command(label="Load Sample (Iris)", command=lambda: self._load_sample("iris"))
        file_menu.add_command(label="Load Sample (Wine)", command=lambda: self._load_sample("wine"))
        file_menu.add_command(label="Load Sample (Breast Cancer)", command=lambda: self._load_sample("breast_cancer"))
        file_menu.add_command(label="Load Sample (Diabetes)", command=lambda: self._load_sample("diabetes"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Cmd+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self._undo, accelerator="Cmd+Z")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About PyWeka", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

        # Keyboard shortcuts
        self.root.bind("<Command-o>", lambda e: self._open_file())
        self.root.bind("<Command-s>", lambda e: self._save_file())
        self.root.bind("<Command-z>", lambda e: self._undo())
        self.root.bind("<Command-q>", lambda e: self.root.quit())
        # Linux/Windows fallback
        self.root.bind("<Control-o>", lambda e: self._open_file())
        self.root.bind("<Control-s>", lambda e: self._save_file())
        self.root.bind("<Control-z>", lambda e: self._undo())

    # ── Toolbar ───────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = ttk.Frame(self.root)
        tb.pack(fill=tk.X, padx=6, pady=(4, 0))
        ttk.Button(tb, text="Open Dataset", command=self._open_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(tb, text="Save Dataset", command=self._save_file).pack(side=tk.LEFT, padx=3)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(tb, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=3)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Sample data buttons
        ttk.Label(tb, text="Samples:").pack(side=tk.LEFT, padx=(0, 3))
        for name in ("iris", "wine", "breast_cancer", "diabetes"):
            ttk.Button(tb, text=name.replace("_", " ").title(),
                       command=lambda n=name: self._load_sample(n)).pack(side=tk.LEFT, padx=2)

        # Dataset info label
        self.dataset_label = ttk.Label(tb, text="No dataset loaded", font=("Helvetica", 10, "italic"))
        self.dataset_label.pack(side=tk.RIGHT, padx=8)

    # ── Tabs ──────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.preprocess = PreprocessTab(self.notebook, self.dm, self._set_status)
        self.classify = ClassifyTab(self.notebook, self.dm, self._set_status)
        self.regression = RegressionTab(self.notebook, self.dm, self._set_status)
        self.cluster = ClusterTab(self.notebook, self.dm, self._set_status)
        self.associate = AssociateTab(self.notebook, self.dm, self._set_status)
        self.select = SelectTab(self.notebook, self.dm, self._set_status)
        self.visualize = VisualizeTab(self.notebook, self.dm, self._set_status)

        self.notebook.add(self.preprocess, text="  Preprocess  ")
        self.notebook.add(self.classify, text="  Classify  ")
        self.notebook.add(self.regression, text="  Regression  ")
        self.notebook.add(self.cluster, text="  Cluster  ")
        self.notebook.add(self.associate, text="  Associate  ")
        self.notebook.add(self.select, text="  Select Attributes  ")
        self.notebook.add(self.visualize, text="  Visualize  ")

        # Update dataset label on data change
        self.dm.add_listener(self._update_dataset_label)

    # ── Status bar ────────────────────────────────────────────────────
    def _build_statusbar(self):
        self.statusbar = StatusBar(self.root)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=2)

    def _set_status(self, msg: str):
        self.statusbar.set(msg)

    def _update_dataset_label(self):
        if self.dm.df is not None:
            s = self.dm.summary()
            self.dataset_label.config(
                text=f"{s['filename']}  ({s['rows']} x {s['cols']})"
            )

    # ── File operations ───────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Dataset",
            filetypes=[
                ("All supported", "*.csv *.tsv *.xlsx *.xls *.arff"),
                ("CSV files", "*.csv"),
                ("TSV files", "*.tsv"),
                ("Excel files", "*.xlsx *.xls"),
                ("ARFF files", "*.arff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            try:
                self.dm.load(path)
                self._set_status(f"Loaded: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _save_file(self):
        if self.dm.df is None:
            messagebox.showinfo("PyWeka", "No dataset to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Dataset",
            defaultextension=".csv",
            filetypes=[
                ("CSV", "*.csv"),
                ("TSV", "*.tsv"),
                ("Excel", "*.xlsx"),
            ],
        )
        if path:
            try:
                self.dm.save(path)
                self._set_status(f"Saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def _undo(self):
        if self.dm.undo():
            self._set_status("Undo successful")
        else:
            messagebox.showinfo("PyWeka", "Nothing to undo.")

    # ── Sample datasets ───────────────────────────────────────────────
    def _load_sample(self, name: str):
        import pandas as pd
        from sklearn import datasets

        loaders = {
            "iris": datasets.load_iris,
            "wine": datasets.load_wine,
            "breast_cancer": datasets.load_breast_cancer,
            "diabetes": datasets.load_diabetes,
        }
        if name not in loaders:
            return
        data = loaders[name]()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if hasattr(data, "target_names") and len(data.target_names) > 0:
            df["target"] = pd.Categorical.from_codes(data.target, data.target_names)
        else:
            df["target"] = data.target
        self.dm.df = df
        self.dm.filename = f"{name}_dataset"
        self.dm._history = [df.copy()]
        self.dm._notify()
        self._set_status(f"Loaded sample: {name} ({len(df)} rows)")

    # ── About ─────────────────────────────────────────────────────────
    def _about(self):
        messagebox.showinfo(
            "About PyWeka",
            "PyWeka – Machine Learning Explorer\n\n"
            "A Python implementation inspired by Weka.\n\n"
            "Features:\n"
            "  - Data Preprocessing\n"
            "  - Classification (13+ algorithms)\n"
            "  - Regression (13+ algorithms)\n"
            "  - Clustering (5 algorithms)\n"
            "  - Association Rules\n"
            "  - Feature Selection\n"
            "  - Interactive Visualization\n\n"
            "Built with scikit-learn, XGBoost,\n"
            "matplotlib, seaborn & Tkinter.",
        )

    # ── Run ───────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()
