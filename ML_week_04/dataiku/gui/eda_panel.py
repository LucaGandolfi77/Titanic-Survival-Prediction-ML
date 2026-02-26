"""
EDA Panel – Exploratory Data Analysis charts and statistics.

Features:
  • Auto-EDA report (summary stats, correlation, distributions, missing values)
  • Manual chart builder (scatter, line, bar, box, histogram, heatmap)
  • Save charts as PNG
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from gui.themes import Colors, FONT_NORMAL, FONT_BOLD, FONT_SMALL


# ---------------------------------------------------------------------------
# Helper: dark style for matplotlib
# ---------------------------------------------------------------------------

def _apply_dark_style() -> None:
    """Set matplotlib rcParams for a dark background matching the GUI."""
    plt.rcParams.update({
        "figure.facecolor": Colors.PANEL,
        "axes.facecolor": Colors.CANVAS_BG,
        "axes.edgecolor": Colors.BORDER,
        "axes.labelcolor": Colors.TEXT,
        "text.color": Colors.TEXT,
        "xtick.color": Colors.TEXT_DIM,
        "ytick.color": Colors.TEXT_DIM,
        "grid.color": Colors.BORDER,
        "legend.facecolor": Colors.PANEL,
        "legend.edgecolor": Colors.BORDER,
    })


# ---------------------------------------------------------------------------
# EDAPanel
# ---------------------------------------------------------------------------

class EDAPanel(ttk.Frame):
    """Tabbed panel for Exploratory Data Analysis."""

    def __init__(
        self,
        parent: tk.Widget,
        get_dataset_names: Callable[[], List[str]],
        get_dataframe: Callable[[str], pd.DataFrame],
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._get_names = get_dataset_names
        self._get_df = get_dataframe
        self._current_fig: Optional[Figure] = None
        self._canvas_widget: Optional[FigureCanvasTkAgg] = None
        _apply_dark_style()
        self._build_ui()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Top bar: dataset selector + auto-EDA button
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(top, text="Dataset:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self._ds_var = tk.StringVar()
        self._ds_combo = ttk.Combobox(top, textvariable=self._ds_var, state="readonly", width=24)
        self._ds_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="\u21bb", width=3, command=self._refresh_datasets).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top, text="Auto EDA", style="Accent.TButton", command=self._auto_eda).pack(side=tk.LEFT, padx=4)

        # Notebook with two tabs: Auto-EDA and Chart Builder
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Auto-EDA tab ---
        self._auto_frame = ttk.Frame(self._nb)
        self._nb.add(self._auto_frame, text="Auto EDA")
        self._auto_canvas_frame = ttk.Frame(self._auto_frame)
        self._auto_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # --- Chart builder tab ---
        self._chart_frame = ttk.Frame(self._nb)
        self._nb.add(self._chart_frame, text="Chart Builder")
        self._build_chart_builder()

    def _build_chart_builder(self) -> None:
        """Build the manual chart builder interface."""
        controls = ttk.Frame(self._chart_frame)
        controls.pack(fill=tk.X, padx=6, pady=4)

        # Chart type
        ttk.Label(controls, text="Chart:").grid(row=0, column=0, padx=2, sticky=tk.W)
        self._chart_type_var = tk.StringVar(value="scatter")
        chart_combo = ttk.Combobox(
            controls, textvariable=self._chart_type_var, state="readonly", width=14,
            values=["scatter", "line", "bar", "box", "histogram", "heatmap"],
        )
        chart_combo.grid(row=0, column=1, padx=4)

        # X axis
        ttk.Label(controls, text="X:").grid(row=0, column=2, padx=2, sticky=tk.W)
        self._x_var = tk.StringVar()
        self._x_combo = ttk.Combobox(controls, textvariable=self._x_var, state="readonly", width=16)
        self._x_combo.grid(row=0, column=3, padx=4)

        # Y axis
        ttk.Label(controls, text="Y:").grid(row=0, column=4, padx=2, sticky=tk.W)
        self._y_var = tk.StringVar()
        self._y_combo = ttk.Combobox(controls, textvariable=self._y_var, state="readonly", width=16)
        self._y_combo.grid(row=0, column=5, padx=4)

        # Color by
        ttk.Label(controls, text="Color:").grid(row=0, column=6, padx=2, sticky=tk.W)
        self._hue_var = tk.StringVar()
        self._hue_combo = ttk.Combobox(controls, textvariable=self._hue_var, state="readonly", width=14)
        self._hue_combo.grid(row=0, column=7, padx=4)

        ttk.Button(controls, text="Plot", style="Accent.TButton", command=self._plot_chart).grid(row=0, column=8, padx=8)
        ttk.Button(controls, text="Save PNG", command=self._save_chart).grid(row=0, column=9, padx=4)

        # Chart area
        self._chart_canvas_frame = ttk.Frame(self._chart_frame)
        self._chart_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Populate columns when dataset changes
        self._ds_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)

    # -- refresh -------------------------------------------------------------

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
        for combo in (self._x_combo, self._y_combo, self._hue_combo):
            combo["values"] = [""] + cols
        if cols:
            self._x_var.set(cols[0])
            self._y_var.set(cols[1] if len(cols) > 1 else cols[0])

    # -- auto EDA ------------------------------------------------------------

    def _auto_eda(self) -> None:
        """Generate an auto-EDA report with multiple subplots."""
        name = self._ds_var.get()
        if not name:
            messagebox.showwarning("EDA", "Select a dataset first.")
            return
        try:
            df = self._get_df(name)
        except Exception as e:
            messagebox.showerror("EDA", str(e))
            return

        _apply_dark_style()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        n_plots = 2 + len(numeric_cols[:6]) + len(cat_cols[:4])
        n_cols = 3
        n_rows = max(1, -(-n_plots // n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        idx = 0

        # 1. Correlation heatmap
        if len(numeric_cols) > 1 and idx < len(axes_flat):
            ax = axes_flat[idx]
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=len(numeric_cols) <= 8, cmap="coolwarm",
                        ax=ax, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax.set_title("Correlation Heatmap", fontsize=10)
            idx += 1

        # 2. Missing values
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                missing.sort_values(ascending=True).plot.barh(ax=ax, color=Colors.WARNING)
                ax.set_title("Missing Values", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color=Colors.TEXT)
                ax.set_title("Missing Values", fontsize=10)
            idx += 1

        # 3. Distribution of numeric cols
        for col in numeric_cols[:6]:
            if idx < len(axes_flat):
                ax = axes_flat[idx]
                df[col].dropna().hist(bins=30, ax=ax, color=Colors.ACCENT, alpha=0.8)
                ax.set_title(f"Dist: {col}", fontsize=9)
                idx += 1

        # 4. Bar charts for categorical cols
        for col in cat_cols[:4]:
            if idx < len(axes_flat):
                ax = axes_flat[idx]
                vc = df[col].value_counts().head(15)
                vc.plot.bar(ax=ax, color=Colors.NODE_DATASET, alpha=0.85)
                ax.set_title(f"Counts: {col}", fontsize=9)
                ax.tick_params(axis="x", rotation=45)
                idx += 1

        # Hide unused axes
        for j in range(idx, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Auto EDA – {name}", fontsize=13, color=Colors.TEXT_BRIGHT, y=1.01)
        fig.tight_layout()

        self._embed_figure(fig, self._auto_canvas_frame)

    # -- manual chart --------------------------------------------------------

    def _plot_chart(self) -> None:
        name = self._ds_var.get()
        if not name:
            messagebox.showwarning("Chart", "Select a dataset first.")
            return
        try:
            df = self._get_df(name)
        except Exception as e:
            messagebox.showerror("Chart", str(e))
            return

        chart_type = self._chart_type_var.get()
        x = self._x_var.get() or None
        y = self._y_var.get() or None
        hue = self._hue_var.get() or None

        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(9, 5))

        try:
            if chart_type == "scatter":
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.7)
            elif chart_type == "line":
                sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif chart_type == "bar":
                if y:
                    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
                else:
                    df[x].value_counts().head(20).plot.bar(ax=ax, color=Colors.ACCENT)
            elif chart_type == "box":
                sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif chart_type == "histogram":
                col = x or y
                if col:
                    df[col].dropna().hist(bins=30, ax=ax, color=Colors.ACCENT, alpha=0.85)
                    ax.set_xlabel(col)
            elif chart_type == "heatmap":
                numeric = df.select_dtypes(include="number")
                sns.heatmap(numeric.corr(), annot=len(numeric.columns) <= 10,
                            cmap="coolwarm", ax=ax, fmt=".2f")

            ax.set_title(f"{chart_type.title()} – {name}", fontsize=11)
            fig.tight_layout()
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color=Colors.ERROR)

        self._current_fig = fig
        self._embed_figure(fig, self._chart_canvas_frame)

    def _save_chart(self) -> None:
        if self._current_fig is None:
            messagebox.showinfo("Save", "No chart to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
        )
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight",
                                       facecolor=Colors.PANEL)
            messagebox.showinfo("Saved", f"Chart saved to {path}")

    # -- embed matplotlib figure in tkinter ----------------------------------

    def _embed_figure(self, fig: Figure, parent: ttk.Frame) -> None:
        """Replace the content of *parent* with a matplotlib canvas."""
        for child in parent.winfo_children():
            child.destroy()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_widget = canvas
