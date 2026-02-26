"""
left_panel.py – The left sidebar with two collapsible sections:
  1. **Repository browser** (saved processes + results + samples)
  2. **Operator palette** (searchable tree grouped by category)
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from engine.operator_base import (
    OpCategory,
    Operator,
    operators_by_category,
)
from engine import repository as repo
from gui.theme import C, F, G


# ═══════════════════════════════════════════════════════════════════════════
# Repository Panel
# ═══════════════════════════════════════════════════════════════════════════

class RepositoryPanel(ttk.LabelFrame):
    """Tree‑view browsing saved processes, results, and samples."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, text="  Repository  ", style="Panel.TLabelframe", **kw)
        self.on_open_process: Optional[Callable] = None

        self.tree = ttk.Treeview(self, show="tree", selectmode="browse",
                                 style="Panel.Treeview")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._my_proc = self.tree.insert("", "end", text="📁 My Processes", open=True)
        self._my_res = self.tree.insert("", "end", text="📁 My Results", open=False)
        self._samples = self.tree.insert("", "end", text="📁 Samples", open=False)

        self.tree.bind("<Double-1>", self._on_double_click)

        self.refresh()

    # ── Public ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        for parent in (self._my_proc, self._my_res, self._samples):
            for child in self.tree.get_children(parent):
                self.tree.delete(child)

        for p in repo.list_saved_processes():
            self.tree.insert(self._my_proc, "end", text=f"📄 {p.stem}",
                             values=(str(p),))

        for name in repo.list_results():
            self.tree.insert(self._my_res, "end", text=f"📊 {name}")

        for p in repo.list_sample_processes():
            self.tree.insert(self._samples, "end", text=f"📄 {p.stem}",
                             values=(str(p),))

    # ── Events ──────────────────────────────────────────────────────────

    def _on_double_click(self, event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        values = self.tree.item(sel[0], "values")
        if values and self.on_open_process:
            from pathlib import Path
            self.on_open_process(Path(values[0]))


# ═══════════════════════════════════════════════════════════════════════════
# Operator Palette Panel
# ═══════════════════════════════════════════════════════════════════════════

_CAT_ICONS = {
    OpCategory.DATA: "📥",
    OpCategory.TRANSFORM: "🔧",
    OpCategory.FEATURE: "🧬",
    OpCategory.MODEL: "🧠",
    OpCategory.EVALUATION: "📊",
    OpCategory.VISUALIZATION: "📈",
    OpCategory.UTILITY: "⚙️",
}


class OperatorPalette(ttk.LabelFrame):
    """Searchable list of available operators grouped by category."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, text="  Operators  ", style="Panel.TLabelframe", **kw)
        self.on_add_operator: Optional[Callable] = None  # callback(op_type: str)

        # Search bar
        search_frame = ttk.Frame(self, style="Panel.TFrame")
        search_frame.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(search_frame, text="🔍", style="Panel.TLabel").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter())
        self._search_entry = ttk.Entry(search_frame, textvariable=self._search_var,
                                       style="TEntry")
        self._search_entry.pack(side="left", fill="x", expand=True, padx=4)

        # Tree
        self.tree = ttk.Treeview(self, show="tree", selectmode="browse",
                                 style="Panel.Treeview")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<Double-1>", self._on_double_click)
        # Drag‑start
        self.tree.bind("<ButtonPress-1>", self._on_press)
        self.tree.bind("<B1-Motion>", self._on_motion)
        self.tree.bind("<ButtonRelease-1>", self._on_release)

        self._drag_data: Optional[str] = None  # op_type being dragged

        self._populate()

    # ── populate tree ───────────────────────────────────────────────────

    def _populate(self, filter_text: str = "") -> None:
        self.tree.delete(*self.tree.get_children())
        groups = operators_by_category()
        ft = filter_text.lower()
        for cat in OpCategory:
            ops = groups.get(cat, [])
            if ft:
                ops = [o for o in ops if ft in o.op_type.lower()]
            if not ops:
                continue
            icon = _CAT_ICONS.get(cat, "📦")
            parent = self.tree.insert("", "end", text=f"{icon} {cat.value}",
                                      open=bool(ft))
            for cls in sorted(ops, key=lambda c: c.op_type):
                self.tree.insert(parent, "end", text=f"  {cls.op_type}",
                                 values=(cls.op_type,))

    def _filter(self) -> None:
        self._populate(self._search_var.get())

    # ── events ──────────────────────────────────────────────────────────

    def _on_double_click(self, event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        if vals and self.on_add_operator:
            self.on_add_operator(vals[0])

    def _on_press(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        if item:
            vals = self.tree.item(item, "values")
            self._drag_data = vals[0] if vals else None

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag_data:
            self.configure(cursor="plus")

    def _on_release(self, event: tk.Event) -> None:
        self.configure(cursor="")
        self._drag_data = None


# ═══════════════════════════════════════════════════════════════════════════
# Composite left panel
# ═══════════════════════════════════════════════════════════════════════════

class LeftPanel(ttk.Frame):
    """Combines the RepositoryPanel and OperatorPalette into a single sidebar."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)

        # Header
        hdr = ttk.Label(self, text="RapidMiner Lite", style="Title.TLabel")
        hdr.pack(fill="x", padx=8, pady=(8, 4))

        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True, padx=4, pady=4)

        self.repository = RepositoryPanel(pw)
        pw.add(self.repository, weight=1)

        self.palette = OperatorPalette(pw)
        pw.add(self.palette, weight=2)
