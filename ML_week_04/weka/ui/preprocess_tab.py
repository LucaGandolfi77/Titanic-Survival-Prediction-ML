"""
preprocess_tab.py – Preprocess panel (mirrors Weka Explorer ▸ Preprocess).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from core.data_manager import DataManager
from core.preprocessor import Preprocessor
from ui.widgets import AttributeList, ScrolledLog, LabeledCombo, LabeledEntry


class PreprocessTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.pp = Preprocessor()
        self.status = status_cb or (lambda m: None)
        self._build_ui()
        dm.add_listener(self._refresh)

    # ── UI layout ─────────────────────────────────────────────────────
    def _build_ui(self):
        # Top: dataset summary
        self.summary_var = tk.StringVar(value="No dataset loaded.")
        ttk.Label(self, textvariable=self.summary_var, font=("Helvetica", 11, "bold"),
                  wraplength=900).pack(fill=tk.X, padx=10, pady=(8, 4))

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: attribute list
        left = ttk.LabelFrame(paned, text="Attributes")
        self.attr_list = AttributeList(left)
        self.attr_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.attr_list.tree.bind("<<TreeviewSelect>>", self._on_attr_select)
        paned.add(left, weight=1)

        # Right: operations + details
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        # Attribute detail
        self.detail_log = ScrolledLog(right, height=8)
        self.detail_log.pack(fill=tk.X, padx=4, pady=4)

        # Filter / operations panel
        ops = ttk.LabelFrame(right, text="Filters / Operations")
        ops.pack(fill=tk.X, padx=4, pady=4)

        r0 = ttk.Frame(ops)
        r0.pack(fill=tk.X, padx=6, pady=3)
        self.col_combo = LabeledCombo(r0, "Column:", [], width=22)
        self.col_combo.pack(side=tk.LEFT, padx=(0, 10))

        # Missing value handling
        r1 = ttk.Frame(ops)
        r1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r1, text="Missing Values:").pack(side=tk.LEFT)
        for strategy in ("mean", "median", "mode", "zero", "drop"):
            ttk.Button(r1, text=strategy.capitalize(),
                       command=lambda s=strategy: self._fill_missing(s)).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Drop All Missing Rows",
                   command=self._drop_all_missing).pack(side=tk.LEFT, padx=6)

        # Encoding
        r2 = ttk.Frame(ops)
        r2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r2, text="Encode:").pack(side=tk.LEFT)
        ttk.Button(r2, text="Label Encode", command=self._label_encode).pack(side=tk.LEFT, padx=2)
        ttk.Button(r2, text="One-Hot Encode", command=self._one_hot_encode).pack(side=tk.LEFT, padx=2)
        ttk.Button(r2, text="Ordinal Encode", command=self._ordinal_encode).pack(side=tk.LEFT, padx=2)

        # Scaling
        r3 = ttk.Frame(ops)
        r3.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r3, text="Scale:").pack(side=tk.LEFT)
        ttk.Button(r3, text="Standardize (Z-score)", command=self._standardize).pack(side=tk.LEFT, padx=2)
        ttk.Button(r3, text="Normalize (0-1)", command=self._normalize).pack(side=tk.LEFT, padx=2)

        # Column ops
        r4 = ttk.Frame(ops)
        r4.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r4, text="Remove Selected Columns", command=self._remove_cols).pack(side=tk.LEFT, padx=2)
        ttk.Button(r4, text="Remove Outliers (IQR)", command=self._remove_outliers).pack(side=tk.LEFT, padx=2)
        self.bins_entry = LabeledEntry(r4, "Discretize bins:", default="5", width=5)
        self.bins_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(r4, text="Discretize", command=self._discretize).pack(side=tk.LEFT, padx=2)

        # Cast type
        r5 = ttk.Frame(ops)
        r5.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r5, text="Cast Type:").pack(side=tk.LEFT)
        for dtype in ("numeric", "string", "category"):
            ttk.Button(r5, text=dtype.capitalize(),
                       command=lambda d=dtype: self._cast(d)).pack(side=tk.LEFT, padx=2)
        ttk.Separator(r5, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(r5, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=4)

        # Sample
        r6 = ttk.Frame(ops)
        r6.pack(fill=tk.X, padx=6, pady=3)
        self.sample_entry = LabeledEntry(r6, "Sample N rows:", default="1000", width=8)
        self.sample_entry.pack(side=tk.LEFT)
        ttk.Button(r6, text="Sample", command=self._sample).pack(side=tk.LEFT, padx=4)
        self.sample_frac = LabeledEntry(r6, "Frac (0-1):", default="0.1", width=6)
        self.sample_frac.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(r6, text="Sample %", command=self._sample_frac).pack(side=tk.LEFT, padx=4)

        # Data preview
        preview_frame = ttk.LabelFrame(right, text="Data Preview (first 50 rows)")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.preview_tree = ttk.Treeview(preview_frame, show="headings", height=6)
        xsb = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        ysb = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Refresh on data change ────────────────────────────────────────
    def _refresh(self):
        df = self.dm.df
        if df is None:
            self.summary_var.set("No dataset loaded.")
            self.attr_list.populate(None)
            return
        s = self.dm.summary()
        self.summary_var.set(
            f"Dataset: {s['filename']}  |  {s['rows']} rows x {s['cols']} cols  |  "
            f"Numeric: {len(s['numeric'])}  Categorical: {len(s['categorical'])}  |  "
            f"Missing: {s['missing']}  |  Memory: {s['mem_mb']} MB"
        )
        self.attr_list.populate(df)
        cols = list(df.columns)
        self.col_combo.set_values(cols, default=cols[0] if cols else None)
        self._update_preview()
        self.status(f"Dataset loaded: {s['rows']} rows x {s['cols']} cols")

    def _update_preview(self):
        df = self.dm.df
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        if df is None:
            return
        cols = list(df.columns)[:30]  # limit columns for performance
        self.preview_tree["columns"] = cols
        for c in cols:
            self.preview_tree.heading(c, text=c)
            self.preview_tree.column(c, width=90, minwidth=50)
        for _, row in df.head(50).iterrows():
            vals = [str(row[c])[:30] if pd.notna(row[c]) else "?" for c in cols]
            self.preview_tree.insert("", tk.END, values=vals)

    def _on_attr_select(self, event=None):
        sel = self.attr_list.get_selected()
        if not sel or self.dm.df is None:
            return
        col = sel[0]
        stats = self.dm.column_stats(col)
        self.detail_log.clear()
        for k, v in stats.items():
            if k == "top_values":
                self.detail_log.append(f"  Top values:")
                for val, cnt in v.items():
                    self.detail_log.append(f"    {val}: {cnt}")
            else:
                self.detail_log.append(f"  {k}: {v}")
        self.col_combo.var.set(col)

    # ── Operations ────────────────────────────────────────────────────
    def _get_col(self) -> str | None:
        col = self.col_combo.get()
        if not col or self.dm.df is None:
            messagebox.showwarning("PyWeka", "Select a column first.")
            return None
        return col

    def _fill_missing(self, strategy: str):
        col = self._get_col()
        if col is None:
            return
        self.dm.set_df(self.pp.fill_missing(self.dm.df, col, strategy))
        self.status(f"Filled missing in '{col}' with {strategy}")

    def _drop_all_missing(self):
        if self.dm.df is None:
            return
        self.dm.set_df(self.pp.remove_missing_rows(self.dm.df))
        self.status("Dropped all rows with missing values")

    def _label_encode(self):
        col = self._get_col()
        if col is None:
            return
        df, _ = self.pp.label_encode(self.dm.df, col)
        self.dm.set_df(df)
        self.status(f"Label-encoded '{col}'")

    def _one_hot_encode(self):
        col = self._get_col()
        if col is None:
            return
        self.dm.set_df(self.pp.one_hot_encode(self.dm.df, col))
        self.status(f"One-hot encoded '{col}'")

    def _ordinal_encode(self):
        col = self._get_col()
        if col is None:
            return
        self.dm.set_df(self.pp.ordinal_encode(self.dm.df, col))
        self.status(f"Ordinal-encoded '{col}'")

    def _standardize(self):
        if self.dm.df is None:
            return
        cols = self.attr_list.get_selected()
        if not cols:
            cols = list(self.dm.df.select_dtypes("number").columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dm.df[c])]
        if not num_cols:
            messagebox.showinfo("PyWeka", "No numeric columns selected.")
            return
        self.dm.set_df(self.pp.standardize(self.dm.df, num_cols))
        self.status(f"Standardized {len(num_cols)} column(s)")

    def _normalize(self):
        if self.dm.df is None:
            return
        cols = self.attr_list.get_selected()
        if not cols:
            cols = list(self.dm.df.select_dtypes("number").columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dm.df[c])]
        if not num_cols:
            messagebox.showinfo("PyWeka", "No numeric columns selected.")
            return
        self.dm.set_df(self.pp.normalize(self.dm.df, num_cols))
        self.status(f"Normalized {len(num_cols)} column(s)")

    def _remove_cols(self):
        if self.dm.df is None:
            return
        sel = self.attr_list.get_selected()
        if not sel:
            messagebox.showinfo("PyWeka", "Select attributes to remove.")
            return
        self.dm.set_df(self.pp.remove_columns(self.dm.df, sel))
        self.status(f"Removed {len(sel)} column(s)")

    def _remove_outliers(self):
        col = self._get_col()
        if col is None or not pd.api.types.is_numeric_dtype(self.dm.df[col]):
            messagebox.showinfo("PyWeka", "Select a numeric column.")
            return
        self.dm.set_df(self.pp.remove_outliers_iqr(self.dm.df, col))
        self.status(f"Removed outliers from '{col}' using IQR")

    def _discretize(self):
        col = self._get_col()
        if col is None or not pd.api.types.is_numeric_dtype(self.dm.df[col]):
            messagebox.showinfo("PyWeka", "Select a numeric column.")
            return
        bins = self.bins_entry.get_int(fallback=5)
        self.dm.set_df(self.pp.discretize(self.dm.df, col, bins=bins))
        self.status(f"Discretized '{col}' into {bins} bins")

    def _cast(self, dtype: str):
        col = self._get_col()
        if col is None:
            return
        self.dm.set_df(self.pp.cast_column(self.dm.df, col, dtype))
        self.status(f"Cast '{col}' to {dtype}")

    def _undo(self):
        if self.dm.undo():
            self.status("Undo successful")
        else:
            messagebox.showinfo("PyWeka", "Nothing to undo.")

    def _sample(self):
        if self.dm.df is None:
            return
        n = self.sample_entry.get_int(fallback=1000)
        self.dm.set_df(self.pp.sample(self.dm.df, n=n))
        self.status(f"Sampled {n} rows")

    def _sample_frac(self):
        if self.dm.df is None:
            return
        frac = self.sample_frac.get_float(fallback=0.1)
        self.dm.set_df(self.pp.sample(self.dm.df, frac=frac))
        self.status(f"Sampled {frac*100:.0f}% of rows")
