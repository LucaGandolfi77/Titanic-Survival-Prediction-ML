"""
results_panel.py – Bottom/centre panel that shows execution results:
  - Data View (paginated table)
  - Statistics
  - Charts (embedded matplotlib)
  - Model Info
  - Log (scrolling text)
"""
from __future__ import annotations

import io
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional

import pandas as pd

from gui.theme import C, F

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════════
# Data View – paginated DataFrame table
# ═══════════════════════════════════════════════════════════════════════════

PAGE_SIZE = 100


class DataView(ttk.Frame):
    """Displays a pandas DataFrame in a treeview with pagination."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self.df: Optional[pd.DataFrame] = None
        self._page: int = 0

        # Treeview
        self._tree_frame = ttk.Frame(self, style="Panel.TFrame")
        self._tree_frame.pack(fill="both", expand=True)
        self._tree = ttk.Treeview(self._tree_frame, show="headings",
                                  selectmode="browse", style="Panel.Treeview")
        xsb = ttk.Scrollbar(self._tree_frame, orient="horizontal",
                            command=self._tree.xview)
        ysb = ttk.Scrollbar(self._tree_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._tree.pack(side="left", fill="both", expand=True)

        # Pagination bar
        bar = ttk.Frame(self, style="Panel.TFrame")
        bar.pack(fill="x", padx=4, pady=2)
        ttk.Button(bar, text="◀ Prev", command=self._prev,
                   style="TButton").pack(side="left")
        self._page_lbl = ttk.Label(bar, text="", style="Panel.TLabel")
        self._page_lbl.pack(side="left", padx=8)
        ttk.Button(bar, text="Next ▶", command=self._next,
                   style="TButton").pack(side="left")
        self._shape_lbl = ttk.Label(bar, text="", style="Dim.TLabel")
        self._shape_lbl.pack(side="right")

    def show(self, df: pd.DataFrame) -> None:
        self.df = df
        self._page = 0
        self._show_page()

    def _show_page(self) -> None:
        t = self._tree
        t.delete(*t.get_children())
        if self.df is None or self.df.empty:
            t["columns"] = ["(empty)"]
            t.heading("(empty)", text="No Data")
            self._page_lbl.configure(text="")
            self._shape_lbl.configure(text="")
            return

        cols = list(self.df.columns)
        t["columns"] = cols
        for c in cols:
            t.heading(c, text=str(c), anchor="w")
            t.column(c, width=100, minwidth=50, stretch=True)

        start = self._page * PAGE_SIZE
        end = start + PAGE_SIZE
        chunk = self.df.iloc[start:end]
        for _, row in chunk.iterrows():
            vals = [str(v) for v in row.values]
            t.insert("", "end", values=vals)

        total_pages = max(1, (len(self.df) - 1) // PAGE_SIZE + 1)
        self._page_lbl.configure(text=f"Page {self._page + 1}/{total_pages}")
        self._shape_lbl.configure(
            text=f"{len(self.df):,} rows × {len(self.df.columns)} cols")

    def _prev(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._show_page()

    def _next(self) -> None:
        if self.df is not None:
            max_page = max(0, (len(self.df) - 1) // PAGE_SIZE)
            if self._page < max_page:
                self._page += 1
                self._show_page()


# ═══════════════════════════════════════════════════════════════════════════
# Statistics View
# ═══════════════════════════════════════════════════════════════════════════

class StatsView(ttk.Frame):
    """Displays df.describe() and dtype info."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self._text = tk.Text(self, bg=C.PANEL, fg=C.TEXT,
                             insertbackground=C.TEXT, font=F.MONO,
                             relief="flat", state="disabled", wrap="none")
        ysb = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=ysb.set)
        self._text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

    def show(self, df: pd.DataFrame) -> None:
        buf = io.StringIO()
        buf.write("═══ Data Types ═══\n")
        buf.write(df.dtypes.to_string())
        buf.write("\n\n═══ Statistics ═══\n")
        buf.write(df.describe(include="all").to_string())
        buf.write(f"\n\n═══ Missing Values ═══\n")
        buf.write(df.isnull().sum().to_string())
        buf.write(f"\n\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        buf.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")

        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", buf.getvalue())
        self._text.configure(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════
# Chart View (embedded matplotlib)
# ═══════════════════════════════════════════════════════════════════════════

class ChartView(ttk.Frame):
    """Embeds matplotlib figures produced by visualisation operators."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self._canvas_widget: Optional[Any] = None

    def show_figure(self, fig: Any) -> None:
        """Display a matplotlib Figure."""
        self._clear()
        if not HAS_MPL or fig is None:
            lbl = ttk.Label(self, text="No chart available.",
                            style="Dim.TLabel")
            lbl.pack(padx=20, pady=30)
            return
        fig.set_facecolor(C.PANEL)
        for ax in fig.get_axes():
            ax.set_facecolor(C.PANEL)
            ax.tick_params(colors=C.TEXT_DIM)
            ax.xaxis.label.set_color(C.TEXT)
            ax.yaxis.label.set_color(C.TEXT)
            ax.title.set_color(C.TEXT)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        self._canvas_widget = canvas.get_tk_widget()
        self._canvas_widget.pack(fill="both", expand=True)

    def _clear(self) -> None:
        if self._canvas_widget:
            self._canvas_widget.destroy()
            self._canvas_widget = None
        for w in self.winfo_children():
            w.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# Model Info View
# ═══════════════════════════════════════════════════════════════════════════

class ModelInfoView(ttk.Frame):
    """Displays model parameters and performance summary."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self._text = tk.Text(self, bg=C.PANEL, fg=C.TEXT,
                             insertbackground=C.TEXT, font=F.MONO,
                             relief="flat", state="disabled", wrap="word")
        ysb = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=ysb.set)
        self._text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

    def show(self, info: Dict[str, Any]) -> None:
        buf = io.StringIO()
        for key, val in info.items():
            buf.write(f"  {key}: {val}\n")
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", buf.getvalue())
        self._text.configure(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════
# Log View
# ═══════════════════════════════════════════════════════════════════════════

class LogView(ttk.Frame):
    """Scrollable text pane for execution log messages."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)
        self._text = tk.Text(self, bg="#0d1117", fg="#c9d1d9",
                             insertbackground=C.TEXT, font=F.MONO,
                             relief="flat", state="disabled", wrap="word")
        ysb = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=ysb.set)
        self._text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        # Tag colours
        self._text.tag_configure("info", foreground="#58a6ff")
        self._text.tag_configure("success", foreground="#3fb950")
        self._text.tag_configure("error", foreground="#f85149")
        self._text.tag_configure("warn", foreground="#d29922")

    def append(self, msg: str, tag: str = "info") -> None:
        self._text.configure(state="normal")
        self._text.insert("end", msg + "\n", tag)
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════
# Composite Results Panel (tabbed)
# ═══════════════════════════════════════════════════════════════════════════

class ResultsPanel(ttk.Frame):
    """Tabbed container holding all result views."""

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Panel.TFrame", **kw)

        self.notebook = ttk.Notebook(self, style="TNotebook")
        self.notebook.pack(fill="both", expand=True)

        self.data_view = DataView(self.notebook)
        self.stats_view = StatsView(self.notebook)
        self.chart_view = ChartView(self.notebook)
        self.model_view = ModelInfoView(self.notebook)
        self.log_view = LogView(self.notebook)

        self.notebook.add(self.data_view, text="  📊 Data  ")
        self.notebook.add(self.stats_view, text="  📈 Statistics  ")
        self.notebook.add(self.chart_view, text="  🎨 Charts  ")
        self.notebook.add(self.model_view, text="  🧠 Model  ")
        self.notebook.add(self.log_view, text="  📝 Log  ")

    # ── convenience helpers ─────────────────────────────────────────────

    def show_dataframe(self, df: pd.DataFrame) -> None:
        self.data_view.show(df)
        self.stats_view.show(df)
        self.notebook.select(0)

    def show_chart(self, fig: Any) -> None:
        self.chart_view.show_figure(fig)
        self.notebook.select(2)

    def show_model_info(self, info: Dict[str, Any]) -> None:
        self.model_view.show(info)
        self.notebook.select(3)

    def log(self, msg: str, tag: str = "info") -> None:
        self.log_view.append(msg, tag)

    def show_results(self, results: Dict[str, Dict]) -> None:
        """Inspect all operator outputs and display the most relevant."""
        for op_id, outputs in results.items():
            for port_name, value in outputs.items():
                if isinstance(value, pd.DataFrame):
                    self.show_dataframe(value)
                elif hasattr(value, "savefig"):  # matplotlib Figure
                    self.show_chart(value)
                elif isinstance(value, dict) and value:
                    self.show_model_info(value)
