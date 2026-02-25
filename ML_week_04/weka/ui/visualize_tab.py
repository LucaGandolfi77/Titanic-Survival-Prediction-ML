"""
visualize_tab.py – Visualization panel (mirrors Weka Explorer ▸ Visualize).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from core.data_manager import DataManager
from ui.widgets import LabeledCombo


class VisualizeTab(ttk.Frame):
    def __init__(self, parent, dm: DataManager, status_cb=None, **kw):
        super().__init__(parent, **kw)
        self.dm = dm
        self.status = status_cb or (lambda m: None)
        self._build_ui()
        dm.add_listener(self._on_data_change)

    def _build_ui(self):
        # ── Controls ──
        ctrl = ttk.LabelFrame(self, text="Plot Settings")
        ctrl.pack(fill=tk.X, padx=8, pady=4)

        r0 = ttk.Frame(ctrl)
        r0.pack(fill=tk.X, padx=6, pady=3)
        self.plot_type = LabeledCombo(r0, "Plot Type:", [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Correlation Heatmap",
            "Pairplot (top 5 numeric)",
            "Class Distribution",
            "Violin Plot",
            "Bar Chart (Categorical)",
            "Distribution Grid",
        ], width=28)
        self.plot_type.pack(side=tk.LEFT, padx=(0, 15))

        self.x_combo = LabeledCombo(r0, "X:", [], width=18)
        self.x_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.y_combo = LabeledCombo(r0, "Y:", [], width=18)
        self.y_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.hue_combo = LabeledCombo(r0, "Hue:", ["(none)"], width=18)
        self.hue_combo.pack(side=tk.LEFT)

        r1 = ttk.Frame(ctrl)
        r1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r1, text="Plot", command=self._plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(r1, text="Clear", command=self._clear_canvas).pack(side=tk.LEFT, padx=4)

        self.bins_var = tk.StringVar(value="30")
        ttk.Label(r1, text="Bins:").pack(side=tk.LEFT, padx=(15, 3))
        ttk.Entry(r1, textvariable=self.bins_var, width=5).pack(side=tk.LEFT)

        self.alpha_var = tk.StringVar(value="0.7")
        ttk.Label(r1, text="Alpha:").pack(side=tk.LEFT, padx=(15, 3))
        ttk.Entry(r1, textvariable=self.alpha_var, width=5).pack(side=tk.LEFT)

        # ── Canvas ──
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._fig_canvas = None

    def _on_data_change(self):
        if self.dm.df is None:
            return
        cols = list(self.dm.df.columns)
        self.x_combo.set_values(cols, default=cols[0] if cols else None)
        self.y_combo.set_values(cols, default=cols[1] if len(cols) > 1 else (cols[0] if cols else None))
        self.hue_combo.set_values(["(none)"] + cols, default="(none)")

    def _clear_canvas(self):
        if self._fig_canvas:
            self._fig_canvas.get_tk_widget().destroy()
            self._fig_canvas = None

    def _draw(self, fig):
        self._clear_canvas()
        self._fig_canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        self._fig_canvas.draw()
        self._fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot(self):
        df = self.dm.df
        if df is None:
            messagebox.showwarning("PyWeka", "Load a dataset first.")
            return

        plot_type = self.plot_type.get()
        x_col = self.x_combo.get()
        y_col = self.y_combo.get()
        hue_col = self.hue_combo.get()
        if hue_col == "(none)":
            hue_col = None

        try:
            bins = int(self.bins_var.get())
        except ValueError:
            bins = 30
        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            alpha = 0.7

        try:
            if plot_type == "Histogram":
                fig, ax = plt.subplots(figsize=(10, 6))
                if pd.api.types.is_numeric_dtype(df[x_col]):
                    if hue_col and hue_col in df.columns:
                        for cat in df[hue_col].dropna().unique()[:10]:
                            sub = df[df[hue_col] == cat][x_col].dropna()
                            ax.hist(sub, bins=bins, alpha=alpha, label=str(cat))
                        ax.legend()
                    else:
                        ax.hist(df[x_col].dropna(), bins=bins, alpha=alpha,
                                color="steelblue", edgecolor="white")
                else:
                    vc = df[x_col].value_counts().head(30)
                    ax.bar(range(len(vc)), vc.values, color="steelblue")
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels(vc.index, rotation=45, ha="right")
                ax.set_title(f"Histogram – {x_col}")
                ax.set_xlabel(x_col); ax.set_ylabel("Count")
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                if hue_col and hue_col in df.columns:
                    cats = df[hue_col].dropna().unique()[:15]
                    data = [df[df[hue_col] == c][x_col].dropna().values for c in cats]
                    bp = ax.boxplot(data, labels=[str(c) for c in cats], patch_artist=True)
                    colors = sns.color_palette("viridis", len(cats))
                    for patch, color in zip(bp["boxes"], colors):
                        patch.set_facecolor(color)
                    ax.tick_params(axis="x", rotation=45)
                else:
                    ax.boxplot(df[x_col].dropna().values, patch_artist=True)
                ax.set_title(f"Box Plot – {x_col}")
                ax.set_ylabel(x_col)
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Scatter Plot":
                fig, ax = plt.subplots(figsize=(10, 7))
                if hue_col and hue_col in df.columns:
                    cats = df[hue_col].dropna().unique()[:10]
                    for cat in cats:
                        sub = df[df[hue_col] == cat]
                        ax.scatter(sub[x_col], sub[y_col], alpha=alpha, s=10, label=str(cat))
                    ax.legend(markerscale=2)
                else:
                    ax.scatter(df[x_col], df[y_col], alpha=alpha * 0.5, s=5, c="teal")
                ax.set_title(f"Scatter – {x_col} vs {y_col}")
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Correlation Heatmap":
                num = df.select_dtypes("number")
                if num.shape[1] < 2:
                    messagebox.showinfo("PyWeka", "Need at least 2 numeric columns.")
                    return
                fig, ax = plt.subplots(figsize=(max(8, num.shape[1] * 0.7),
                                                max(6, num.shape[1] * 0.6)))
                corr = num.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=num.shape[1] <= 15,
                            fmt=".2f", cmap="coolwarm", ax=ax,
                            linewidths=0.5, vmin=-1, vmax=1,
                            annot_kws={"size": 7} if num.shape[1] <= 15 else {})
                ax.set_title("Correlation Heatmap")
                plt.tight_layout()
                self._draw(fig)

            elif plot_type.startswith("Pairplot"):
                num_cols = list(df.select_dtypes("number").columns[:5])
                if len(num_cols) < 2:
                    messagebox.showinfo("PyWeka", "Need at least 2 numeric columns.")
                    return
                sub = df[num_cols].dropna().sample(min(1000, len(df)), random_state=42)
                fig = sns.pairplot(sub, diag_kind="hist", plot_kws={"alpha": 0.3, "s": 8}).figure
                fig.suptitle("Pairplot (top 5 numeric)", y=1.01)
                self._draw(fig)

            elif plot_type == "Class Distribution":
                fig, ax = plt.subplots(figsize=(10, 6))
                vc = df[x_col].value_counts().head(20)
                vc.plot.bar(ax=ax, color=sns.color_palette("viridis", len(vc)))
                ax.set_title(f"Class Distribution – {x_col}")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Violin Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                if hue_col and hue_col in df.columns:
                    cats = df[hue_col].dropna().unique()[:10]
                    sub = df[df[hue_col].isin(cats)]
                    sns.violinplot(data=sub, x=hue_col, y=x_col, ax=ax, inner="quartile")
                    ax.tick_params(axis="x", rotation=45)
                else:
                    sns.violinplot(data=df, y=x_col, ax=ax, inner="quartile")
                ax.set_title(f"Violin Plot – {x_col}")
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Bar Chart (Categorical)":
                fig, ax = plt.subplots(figsize=(10, 6))
                vc = df[x_col].value_counts().head(25)
                vc.plot.barh(ax=ax, color=sns.color_palette("rocket", len(vc)))
                ax.set_title(f"Bar Chart – {x_col}")
                ax.set_xlabel("Count"); ax.invert_yaxis()
                plt.tight_layout()
                self._draw(fig)

            elif plot_type == "Distribution Grid":
                num_cols = list(df.select_dtypes("number").columns[:12])
                n = len(num_cols)
                if n == 0:
                    messagebox.showinfo("PyWeka", "No numeric columns.")
                    return
                cols_grid = min(n, 4)
                rows_grid = (n + cols_grid - 1) // cols_grid
                fig, axes = plt.subplots(rows_grid, cols_grid,
                                         figsize=(4 * cols_grid, 3 * rows_grid))
                if n == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                for i, col in enumerate(num_cols):
                    axes[i].hist(df[col].dropna(), bins=bins, alpha=alpha,
                                 color="steelblue", edgecolor="white")
                    axes[i].set_title(col, fontsize=9)
                for i in range(n, len(axes)):
                    axes[i].axis("off")
                plt.suptitle("Distribution Grid", fontsize=13)
                plt.tight_layout()
                self._draw(fig)

        except Exception as e:
            messagebox.showerror("PyWeka", f"Plot error: {e}")
