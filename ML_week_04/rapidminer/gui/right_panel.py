"""
right_panel.py – The right sidebar that shows the parameters of the
currently selected operator and a description pane.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Optional

from engine.operator_base import (
    Operator,
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
)
from gui.theme import C, F


# ═══════════════════════════════════════════════════════════════════════════
# RightPanel
# ═══════════════════════════════════════════════════════════════════════════

class RightPanel(ttk.Frame):
    """Shows parameters + description of the currently selected operator."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self.operator: Optional[Operator] = None
        self.on_params_changed: Optional[Callable] = None
        self._widgets: List[tk.Widget] = []

        # ── title label ─────────────────────────────────────────────────
        self._title = ttk.Label(self, text="Parameters",
                                style="Heading.TLabel")
        self._title.pack(fill="x", padx=8, pady=(8, 4))

        # ── scrollable area ─────────────────────────────────────────────
        container = ttk.Frame(self, style="Panel.TFrame")
        container.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(container, bg=C.PANEL, highlightthickness=0,
                                 width=260)
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=self._canvas.yview)
        self._inner = ttk.Frame(self._canvas, style="Panel.TFrame")
        self._inner.bind("<Configure>",
                         lambda e: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # ── description ─────────────────────────────────────────────────
        self._desc_label = ttk.Label(self, text="", style="Dim.TLabel",
                                     wraplength=250, justify="left")
        self._desc_label.pack(fill="x", padx=8, pady=(4, 8))

        # ── apply / reset ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self, style="Panel.TFrame")
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Apply", command=self._apply,
                   style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Reset", command=self._reset,
                   style="TButton").pack(side="left", padx=2)

        self._show_placeholder()

    # ── public API ──────────────────────────────────────────────────────

    def show_operator(self, op: Optional[Operator]) -> None:
        """Display parameter widgets for *op* (or placeholder if None)."""
        self.operator = op
        self._clear()
        if op is None:
            self._show_placeholder()
            return

        self._title.configure(text=f"⚙ {op.display_name}")
        self._desc_label.configure(text=op.description or "No description.")

        # Operator rename
        self._add_field("Display name", ParamSpec("__display_name__",
                        ParamKind.STRING, default=op.display_name,
                        description="Rename this operator instance."),
                        initial=op.display_name)

        # Real params
        for ps in op.param_specs:
            val = op.params.get(ps.name, ps.default)
            self._add_field(ps.name, ps, initial=val)

    # ── internal ────────────────────────────────────────────────────────

    def _clear(self) -> None:
        for w in self._widgets:
            w.destroy()
        self._widgets.clear()

    def _show_placeholder(self) -> None:
        self._title.configure(text="Parameters")
        self._desc_label.configure(text="Select an operator on the canvas\n"
                                        "to edit its parameters.")
        lbl = ttk.Label(self._inner, text="No operator selected",
                        style="Dim.TLabel")
        lbl.pack(padx=10, pady=30)
        self._widgets.append(lbl)

    def _add_field(self, label: str, spec: ParamSpec, initial: Any = None) -> None:
        # Label
        lbl = ttk.Label(self._inner, text=label, style="Panel.TLabel",
                        font=F.BOLD)
        lbl.pack(fill="x", padx=8, pady=(6, 1), anchor="w")
        self._widgets.append(lbl)

        # Tooltip (description)
        if spec.description:
            tip = ttk.Label(self._inner, text=spec.description,
                            style="Dim.TLabel", font=F.SMALL,
                            wraplength=230)
            tip.pack(fill="x", padx=12, anchor="w")
            self._widgets.append(tip)

        var: Any = None
        widget: tk.Widget

        if spec.kind == ParamKind.BOOL:
            var = tk.BooleanVar(value=bool(initial))
            widget = ttk.Checkbutton(self._inner, variable=var,
                                     style="TCheckbutton")
        elif spec.kind == ParamKind.CHOICE:
            var = tk.StringVar(value=str(initial or ""))
            widget = ttk.Combobox(self._inner, textvariable=var,
                                  values=spec.choices or [],
                                  state="readonly", style="TCombobox")
        elif spec.kind == ParamKind.INT:
            var = tk.IntVar(value=int(initial or 0))
            widget = ttk.Spinbox(self._inner, from_=spec.min_val or 0,
                                 to=spec.max_val or 99999,
                                 textvariable=var, style="TSpinbox")
        elif spec.kind == ParamKind.FLOAT:
            var = tk.DoubleVar(value=float(initial or 0.0))
            widget = ttk.Spinbox(self._inner, from_=spec.min_val or 0.0,
                                 to=spec.max_val or 99999.0,
                                 increment=0.01,
                                 textvariable=var, style="TSpinbox")
        elif spec.kind == ParamKind.FILEPATH:
            var = tk.StringVar(value=str(initial or ""))
            frame = ttk.Frame(self._inner, style="Panel.TFrame")
            entry = ttk.Entry(frame, textvariable=var, style="TEntry")
            entry.pack(side="left", fill="x", expand=True)
            btn = ttk.Button(frame, text="…",
                             command=lambda v=var: self._browse_file(v),
                             style="TButton", width=3)
            btn.pack(side="right", padx=2)
            widget = frame
        elif spec.kind == ParamKind.TEXT:
            widget = tk.Text(self._inner, height=3, bg=C.PANEL,
                             fg=C.TEXT, insertbackground=C.TEXT,
                             font=F.MONO, relief="flat",
                             highlightbackground=C.BORDER,
                             highlightthickness=1)
            widget.insert("1.0", str(initial or ""))
            var = widget  # special case
        elif spec.kind in (ParamKind.COLUMN, ParamKind.COLUMN_LIST):
            var = tk.StringVar(value=str(initial or ""))
            widget = ttk.Entry(self._inner, textvariable=var, style="TEntry")
        else:  # STRING / fallback
            var = tk.StringVar(value=str(initial or ""))
            widget = ttk.Entry(self._inner, textvariable=var, style="TEntry")

        widget.pack(fill="x", padx=8, pady=(0, 2))
        self._widgets.append(widget)

        # Store reference for _apply
        widget._rm_param_name = spec.name  # type: ignore[attr-defined]
        widget._rm_var = var  # type: ignore[attr-defined]
        widget._rm_kind = spec.kind  # type: ignore[attr-defined]

    # ── apply / reset ───────────────────────────────────────────────────

    def _apply(self) -> None:
        if not self.operator:
            return
        for w in self._widgets:
            name = getattr(w, "_rm_param_name", None)
            if name is None:
                continue
            var = getattr(w, "_rm_var", None)
            kind = getattr(w, "_rm_kind", None)
            if var is None:
                continue

            if kind == ParamKind.TEXT and isinstance(var, tk.Text):
                value = var.get("1.0", "end-1c")
            else:
                try:
                    value = var.get()
                except Exception:
                    continue

            if name == "__display_name__":
                self.operator.display_name = str(value)
            else:
                self.operator.set_param(name, value)

        if self.on_params_changed:
            self.on_params_changed(self.operator)

    def _reset(self) -> None:
        if self.operator:
            self.show_operator(self.operator)

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)
