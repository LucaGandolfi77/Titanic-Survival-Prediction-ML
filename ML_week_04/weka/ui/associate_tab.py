"""
associate_tab.py – Association Rules panel (mirrors Weka Explorer ▸ Associate).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

from core.data_manager import DataManager
from ui.widgets import ScrolledLog, LabeledCombo, LabeledEntry


class AssociateTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Left ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        info = ttk.LabelFrame(left, text="Association Rules (Apriori)")
        info.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(info, text="Find frequent patterns and association rules\n"
                  "in categorical / binary data.\n\n"
                  "Data should contain binary (0/1) or categorical columns.\n"
                  "Numeric columns will be auto-discretized into quartiles.",
                  wraplength=300, justify=tk.LEFT).pack(padx=6, pady=6)

        # Parameters
        pf = ttk.LabelFrame(left, text="Parameters")
        pf.pack(fill=tk.X, padx=4, pady=4)
        self.min_support = LabeledEntry(pf, "Min Support:", default="0.1", width=6)
        self.min_support.pack(fill=tk.X, padx=6, pady=3)
        self.min_confidence = LabeledEntry(pf, "Min Confidence:", default="0.5", width=6)
        self.min_confidence.pack(fill=tk.X, padx=6, pady=3)
        self.max_rules = LabeledEntry(pf, "Max Rules:", default="20", width=6)
        self.max_rules.pack(fill=tk.X, padx=6, pady=3)

        self.metric_combo = LabeledCombo(pf, "Sort Metric:",
                                          ["confidence", "lift", "support", "leverage"],
                                          default="lift", width=14)
        self.metric_combo.pack(fill=tk.X, padx=6, pady=3)

        # Feature selection
        feat_frame = ttk.LabelFrame(left, text="Columns to Use")
        feat_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        canvas = tk.Canvas(feat_frame, highlightthickness=0, height=120)
        vsb = ttk.Scrollbar(feat_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.feat_inner = ttk.Frame(canvas)
        self.feat_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.feat_inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.feat_vars: dict[str, tk.BooleanVar] = {}

        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(bf, text="Find Rules", command=self._run).pack(side=tk.LEFT, padx=4)
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
        for col in self.dm.df.columns:
            var = tk.BooleanVar(value=True)
            self.feat_vars[col] = var
            ttk.Checkbutton(self.feat_inner, text=col, variable=var).pack(anchor=tk.W, padx=6, pady=1)

    def _run(self):
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            self.result_log.append("ERROR: mlxtend is not installed.")
            self.result_log.append("Install it with: pip install mlxtend")
            self.result_log.append("\nFalling back to simple co-occurrence analysis...")
            self._fallback_analysis()
            return

        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return

        selected = [c for c, v in self.feat_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("PyWeka", "Select columns.")
            return

        min_sup = self.min_support.get_float(0.1)
        min_conf = self.min_confidence.get_float(0.5)
        max_rules_n = self.max_rules.get_int(20)
        metric = self.metric_combo.get()

        self.status("Mining association rules...")
        self.update_idletasks()

        sub = df[selected].copy()

        # Binarize: one-hot encode all columns
        for col in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[col]):
                sub[col] = pd.qcut(sub[col], q=4, duplicates="drop").astype(str)

        encoded = pd.get_dummies(sub)
        # Convert to bool
        encoded = encoded.astype(bool)

        self.result_log.append("=" * 70)
        self.result_log.append(f"Association Rules  |  Support >= {min_sup}  |  "
                               f"Confidence >= {min_conf}")
        self.result_log.append(f"Encoded columns: {encoded.shape[1]}  |  Rows: {len(encoded)}")
        self.result_log.append("=" * 70)

        try:
            freq = apriori(encoded, min_support=min_sup, use_colnames=True)
            if freq.empty:
                self.result_log.append("No frequent itemsets found. Try lowering min support.")
                return

            self.result_log.append(f"\nFrequent itemsets found: {len(freq)}")

            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            if rules.empty:
                self.result_log.append("No rules found. Try lowering min confidence.")
                return

            rules = rules.sort_values(metric, ascending=False).head(max_rules_n)
            self.result_log.append(f"Association rules found: {len(rules)}\n")

            self.result_log.append(f"{'#':<4} {'Antecedents':<35s} {'Consequents':<35s} "
                                   f"{'Support':>8s} {'Confidence':>10s} {'Lift':>8s}")
            self.result_log.append("-" * 105)

            for i, (_, row) in enumerate(rules.iterrows(), 1):
                ant = ", ".join(list(row["antecedents"])[:3])
                con = ", ".join(list(row["consequents"])[:3])
                self.result_log.append(
                    f"{i:<4} {ant:<35s} {con:<35s} "
                    f"{row['support']:>8.4f} {row['confidence']:>10.4f} {row['lift']:>8.4f}"
                )
            self.result_log.append("")

        except Exception as e:
            self.result_log.append(f"ERROR: {e}")

        self.status("Association rules complete.")

    def _fallback_analysis(self):
        """Simple co-occurrence analysis when mlxtend is not available."""
        df = self.dm.df
        if df is None:
            return

        selected = [c for c, v in self.feat_vars.items() if v.get()]
        cat_cols = [c for c in selected if not pd.api.types.is_numeric_dtype(df[c])]

        if len(cat_cols) < 2:
            self.result_log.append("Need at least 2 categorical columns for co-occurrence analysis.")
            return

        self.result_log.append("\n" + "=" * 60)
        self.result_log.append("Co-occurrence Analysis (Contingency Tables)")
        self.result_log.append("=" * 60)

        # Show top co-occurrences for pairs
        from itertools import combinations
        for c1, c2 in list(combinations(cat_cols[:5], 2)):
            ct = pd.crosstab(df[c1], df[c2])
            # Find top co-occurrences
            top_pairs = []
            for r in ct.index:
                for c in ct.columns:
                    top_pairs.append((r, c, ct.loc[r, c]))
            top_pairs.sort(key=lambda x: x[2], reverse=True)

            self.result_log.append(f"\n--- {c1} x {c2} (top 5 co-occurrences) ---")
            for val1, val2, count in top_pairs[:5]:
                self.result_log.append(f"  {val1} + {val2}: {count}")

    def _clear(self):
        self.result_log.clear()
