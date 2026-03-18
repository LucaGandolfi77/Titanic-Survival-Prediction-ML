"""
widgets.py – Reusable Tkinter widgets for the PyWeka UI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext


class ScrolledLog(ttk.Frame):
    """A scrolled text area for log / result output."""

    def __init__(self, parent, height: int = 20, **kw):
        super().__init__(parent, **kw)
        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, font=("Courier", 11),
            bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
            selectbackground="#585b70", relief=tk.FLAT, padx=8, pady=8,
            height=height,
        )
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.config(state=tk.DISABLED)

    def append(self, msg: str) -> None:
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def clear(self) -> None:
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.config(state=tk.DISABLED)

    def get_text(self) -> str:
        return self.text.get("1.0", tk.END)


class LabeledCombo(ttk.Frame):
    """A label + combobox pair."""

    def __init__(self, parent, label: str, values: list[str],
                 default: str | None = None, width: int = 28, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=label).pack(side=tk.LEFT, padx=(0, 6))
        self.var = tk.StringVar(value=default or (values[0] if values else ""))
        self.combo = ttk.Combobox(self, textvariable=self.var, values=values,
                                  state="readonly", width=width)
        self.combo.pack(side=tk.LEFT)

    def get(self) -> str:
        return self.var.get()

    def set_values(self, values: list[str], default: str | None = None) -> None:
        self.combo["values"] = values
        if default:
            self.var.set(default)
        elif values:
            self.var.set(values[0])


class LabeledEntry(ttk.Frame):
    """A label + entry pair."""

    def __init__(self, parent, label: str, default: str = "",
                 width: int = 10, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=label).pack(side=tk.LEFT, padx=(0, 6))
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side=tk.LEFT)

    def get(self) -> str:
        return self.var.get()

    def get_float(self, fallback: float = 0.0) -> float:
        try:
            return float(self.var.get())
        except ValueError:
            return fallback

    def get_int(self, fallback: int = 0) -> int:
        try:
            return int(self.var.get())
        except ValueError:
            return fallback


class CheckGroup(ttk.LabelFrame):
    """A group of checkbuttons."""

    def __init__(self, parent, text: str, items: list[str],
                 default_checked: bool = True, **kw):
        super().__init__(parent, text=text, **kw)
        self.vars: dict[str, tk.BooleanVar] = {}
        for item in items:
            var = tk.BooleanVar(value=default_checked)
            self.vars[item] = var
            ttk.Checkbutton(self, text=item, variable=var).pack(
                anchor=tk.W, padx=6, pady=1
            )

    def get_selected(self) -> list[str]:
        return [k for k, v in self.vars.items() if v.get()]

    def set_items(self, items: list[str], default_checked: bool = True) -> None:
        for w in self.winfo_children():
            w.destroy()
        self.vars.clear()
        for item in items:
            var = tk.BooleanVar(value=default_checked)
            self.vars[item] = var
            ttk.Checkbutton(self, text=item, variable=var).pack(
                anchor=tk.W, padx=6, pady=1
            )


class ToolBar(ttk.Frame):
    """Horizontal toolbar with buttons."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)

    def add_button(self, text: str, command, **kw) -> ttk.Button:
        btn = ttk.Button(self, text=text, command=command, **kw)
        btn.pack(side=tk.LEFT, padx=3, pady=3)
        return btn

    def add_separator(self) -> None:
        ttk.Separator(self, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=3
        )


class StatusBar(ttk.Frame):
    """Bottom status bar."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.label = ttk.Label(self, text="Ready", anchor=tk.W)
        self.label.pack(fill=tk.X, padx=6, pady=2)

    def set(self, msg: str) -> None:
        self.label.config(text=msg)
        self.update_idletasks()


class AttributeList(ttk.Frame):
    """Scrollable list for dataset attributes (like Weka's attribute panel)."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.tree = ttk.Treeview(
            self, columns=("dtype", "missing", "unique"), show="headings",
            height=12, selectmode="extended",
        )
        self.tree.heading("dtype", text="Type")
        self.tree.heading("missing", text="Missing")
        self.tree.heading("unique", text="Unique")
        self.tree.column("dtype", width=80)
        self.tree.column("missing", width=65)
        self.tree.column("unique", width=65)

        # Add a column for the attribute name
        self.tree["columns"] = ("name", "dtype", "missing", "unique")
        self.tree.heading("name", text="Attribute")
        self.tree.column("name", width=160)

        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def populate(self, df) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        if df is None:
            return
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            unique = int(df[col].nunique())
            self.tree.insert("", tk.END, values=(col, dtype, missing, unique))

    def get_selected(self) -> list[str]:
        return [self.tree.item(i)["values"][0] for i in self.tree.selection()]

    def get_all(self) -> list[str]:
        return [self.tree.item(i)["values"][0] for i in self.tree.get_children()]
