#!/usr/bin/env python3
"""
DataikuLite DSS – main application entry point.

Assembles the full GUI: menu bar, project explorer sidebar, tabbed
workspace (Flow, Datasets, EDA, ML Lab, Notebook), right inspector,
and status bar.
"""
from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path so relative imports work
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import numpy as np
import pandas as pd

from core.project import ProjectManager, Project
from core.dataset import DatasetManager
from core.recipe import RecipeEngine, new_prepare_step, new_recipe_config, STEP_HANDLERS
from core.ml import MLLab
from core.notebook import NotebookManager, NotebookCell
from gui.themes import Colors, FONT_BOLD, FONT_CODE, FONT_HEADING, FONT_NORMAL, FONT_SMALL, FONT_TITLE, apply_dark_theme
from gui.flow_canvas import FlowCanvas, FlowNode
from gui.eda_panel import EDAPanel
from gui.ml_panel import MLPanel
from utils.helpers import (
    SUPPORTED_EXTENSIONS,
    column_stats,
    dataframe_memory,
    run_in_background,
    timestamp_str,
)


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class DataikuLiteApp:
    """Main application window."""

    TITLE = "DataikuLite DSS"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(self.TITLE)
        self.root.geometry("1400x850")
        self.root.minsize(900, 600)

        # Apply dark theme
        self._style = apply_dark_theme(self.root)

        # Core managers
        self.project_mgr = ProjectManager()
        self.dataset_mgr = DatasetManager()
        self.recipe_engine = RecipeEngine()
        self.notebook_mgr = NotebookManager()

        # State
        self._current_dataset: Optional[str] = None

        # Build UI
        self._build_menu()
        self._build_layout()
        self._build_statusbar()

        # Initial project
        self._new_project_silent("Untitled")

    # ======================================================================
    # MENU BAR
    # ======================================================================

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root, bg=Colors.PANEL, fg=Colors.TEXT,
                          activebackground=Colors.ACCENT, activeforeground=Colors.TEXT_BRIGHT,
                          relief=tk.FLAT)

        # File
        file_menu = tk.Menu(menubar, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                            activebackground=Colors.ACCENT)
        file_menu.add_command(label="Import Dataset…", command=self._import_dataset)
        file_menu.add_command(label="Export Dataset…", command=self._export_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Project
        proj_menu = tk.Menu(menubar, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                            activebackground=Colors.ACCENT)
        proj_menu.add_command(label="New Project", command=self._new_project)
        proj_menu.add_command(label="Open Project…", command=self._open_project)
        proj_menu.add_command(label="Save Project", command=self._save_project)
        menubar.add_cascade(label="Project", menu=proj_menu)

        # Run
        run_menu = tk.Menu(menubar, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                           activebackground=Colors.ACCENT)
        run_menu.add_command(label="Run Pipeline", command=self._run_pipeline)
        menubar.add_cascade(label="Run", menu=run_menu)

        # Tools
        tools_menu = tk.Menu(menubar, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                             activebackground=Colors.ACCENT)
        tools_menu.add_command(label="Reset Notebook Namespace", command=self._reset_notebook_ns)
        tools_menu.add_command(label="Export Notebook as .py", command=self._export_notebook)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                            activebackground=Colors.ACCENT)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # ======================================================================
    # LAYOUT
    # ======================================================================

    def _build_layout(self) -> None:
        """Create the three-panel layout: sidebar | centre tabs | inspector."""
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # --- LEFT SIDEBAR (project tree) ---
        self._sidebar = ttk.Frame(main_pw, width=210)
        main_pw.add(self._sidebar, weight=0)
        self._build_sidebar()

        # --- CENTRE (tabs) ---
        centre = ttk.Frame(main_pw)
        main_pw.add(centre, weight=4)
        self._notebook = ttk.Notebook(centre)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._build_tabs()

        # --- RIGHT INSPECTOR ---
        self._inspector = ttk.Frame(main_pw, width=260)
        main_pw.add(self._inspector, weight=0)
        self._build_inspector()

    # -- sidebar -------------------------------------------------------------

    def _build_sidebar(self) -> None:
        ttk.Label(self._sidebar, text="Project Explorer", style="Heading.TLabel").pack(
            padx=8, pady=(8, 4), anchor=tk.W)

        self._tree = ttk.Treeview(self._sidebar, show="tree", selectmode="browse")
        self._tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        btn_row = ttk.Frame(self._sidebar)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="+ Import", command=self._import_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="\u21bb", width=3, command=self._refresh_tree).pack(side=tk.LEFT, padx=2)

    def _refresh_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        # Datasets
        ds_node = self._tree.insert("", tk.END, text="\U0001f4c1 Datasets", open=True)
        for name in self.dataset_mgr.list_names():
            self._tree.insert(ds_node, tk.END, text=f"\U0001f4c4 {name}", values=(name,))
        # Models
        mdl_node = self._tree.insert("", tk.END, text="\U0001f916 Models", open=True)
        # Notebook
        nb_node = self._tree.insert("", tk.END, text="\U0001f4d3 Notebook", open=False)

    def _on_tree_select(self, _event: Any) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self._tree.item(item, "values")
        if vals:
            name = vals[0]
            if self.dataset_mgr.has(name):
                self._current_dataset = name
                self._show_dataset_preview(name)
                self._show_inspector_for_dataset(name)

    # -- centre tabs ---------------------------------------------------------

    def _build_tabs(self) -> None:
        # 1. Flow
        self._flow_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._flow_frame, text="  Flow  ")
        self._flow_canvas = FlowCanvas(
            self._flow_frame,
            on_node_select=self._on_flow_node_select,
            on_run_node=self._on_flow_run_node,
            on_run_all=self._run_pipeline,
            on_configure_node=self._on_flow_configure_node,
        )
        self._flow_canvas.pack(fill=tk.BOTH, expand=True)

        # 2. Datasets
        self._datasets_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._datasets_frame, text="  Datasets  ")
        self._build_datasets_tab()

        # 3. EDA
        self._eda_panel = EDAPanel(
            self._notebook,
            get_dataset_names=self.dataset_mgr.list_names,
            get_dataframe=self.dataset_mgr.get_df,
        )
        self._notebook.add(self._eda_panel, text="  EDA  ")

        # 4. ML Lab
        self._ml_panel = MLPanel(
            self._notebook,
            get_dataset_names=self.dataset_mgr.list_names,
            get_dataframe=self.dataset_mgr.get_df,
            set_status=self._set_status,
        )
        self._notebook.add(self._ml_panel, text="  ML Lab  ")

        # 5. Notebook
        self._nb_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._nb_frame, text="  Notebook  ")
        self._build_notebook_tab()

        # 6. Prepare
        self._prepare_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._prepare_frame, text="  Prepare  ")
        self._build_prepare_tab()

    # -- datasets tab --------------------------------------------------------

    def _build_datasets_tab(self) -> None:
        top = ttk.Frame(self._datasets_frame)
        top.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(top, text="Dataset:", style="Panel.TLabel").pack(side=tk.LEFT)
        self._dt_var = tk.StringVar()
        self._dt_combo = ttk.Combobox(top, textvariable=self._dt_var, state="readonly", width=24)
        self._dt_combo.pack(side=tk.LEFT, padx=4)
        self._dt_combo.bind("<<ComboboxSelected>>", lambda e: self._show_dataset_preview(self._dt_var.get()))
        ttk.Button(top, text="\u21bb", width=3, command=self._refresh_dt_combo).pack(side=tk.LEFT)

        # Pagination
        self._page_var = tk.IntVar(value=0)
        self._total_pages_var = tk.IntVar(value=1)
        page_frame = ttk.Frame(top)
        page_frame.pack(side=tk.RIGHT, padx=8)
        ttk.Button(page_frame, text="◀", width=3, command=self._prev_page).pack(side=tk.LEFT)
        self._page_label = ttk.Label(page_frame, text="Page 1/1", style="Panel.TLabel")
        self._page_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(page_frame, text="▶", width=3, command=self._next_page).pack(side=tk.LEFT)

        # Table
        tree_frame = ttk.Frame(self._datasets_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._data_tree = ttk.Treeview(tree_frame, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._data_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self._data_tree.xview)
        self._data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._data_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    def _refresh_dt_combo(self) -> None:
        names = self.dataset_mgr.list_names()
        self._dt_combo["values"] = names
        if names and not self._dt_var.get():
            self._dt_var.set(names[0])
            self._show_dataset_preview(names[0])

    def _show_dataset_preview(self, name: str) -> None:
        if not name or not self.dataset_mgr.has(name):
            return
        page = self._page_var.get()
        page_df, total_pages = self.dataset_mgr.preview(name, page=page, page_size=100)
        self._total_pages_var.set(total_pages)
        self._page_label.config(text=f"Page {page + 1}/{total_pages}")

        # Rebuild treeview columns
        self._data_tree.delete(*self._data_tree.get_children())
        cols = list(page_df.columns)
        self._data_tree["columns"] = cols
        for c in cols:
            self._data_tree.heading(c, text=c)
            max_w = max(80, min(200, len(c) * 10))
            self._data_tree.column(c, width=max_w, minwidth=60)

        for _, row in page_df.iterrows():
            vals = [str(v) for v in row.values]
            self._data_tree.insert("", tk.END, values=vals)

        info = self.dataset_mgr.get_info(name)
        self._set_status(f"{name}: {info.row_count} rows × {info.col_count} cols | {info.memory}")

    def _prev_page(self) -> None:
        p = max(0, self._page_var.get() - 1)
        self._page_var.set(p)
        name = self._dt_var.get() or self._current_dataset
        if name:
            self._show_dataset_preview(name)

    def _next_page(self) -> None:
        p = min(self._total_pages_var.get() - 1, self._page_var.get() + 1)
        self._page_var.set(p)
        name = self._dt_var.get() or self._current_dataset
        if name:
            self._show_dataset_preview(name)

    # -- prepare tab ---------------------------------------------------------

    def _build_prepare_tab(self) -> None:
        """Prepare recipe tab with step builder and live preview."""
        top = ttk.Frame(self._prepare_frame)
        top.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(top, text="Input Dataset:").pack(side=tk.LEFT)
        self._prep_ds_var = tk.StringVar()
        self._prep_ds_combo = ttk.Combobox(top, textvariable=self._prep_ds_var, state="readonly", width=22)
        self._prep_ds_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="\u21bb", width=3, command=self._refresh_prep_combos).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top, text="Run Recipe", style="Accent.TButton", command=self._run_prepare_recipe).pack(side=tk.LEFT, padx=4)

        # Steps list + add step controls
        pw = ttk.PanedWindow(self._prepare_frame, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: steps list
        steps_frame = ttk.LabelFrame(pw, text="Steps")
        pw.add(steps_frame, weight=1)

        self._steps_listbox = tk.Listbox(
            steps_frame, bg=Colors.INPUT_BG, fg=Colors.TEXT,
            selectbackground=Colors.ACCENT, font=FONT_NORMAL, height=10,
        )
        self._steps_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        step_btns = ttk.Frame(steps_frame)
        step_btns.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(step_btns, text="+ Add Step", command=self._add_prepare_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(step_btns, text="Remove", command=self._remove_prepare_step).pack(side=tk.LEFT, padx=2)

        # Right: step config + preview
        config_frame = ttk.LabelFrame(pw, text="Step Config")
        pw.add(config_frame, weight=2)

        # Step type selector
        st_row = ttk.Frame(config_frame)
        st_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(st_row, text="Step type:").pack(side=tk.LEFT)
        self._step_type_var = tk.StringVar(value="filter_rows")
        step_type_combo = ttk.Combobox(
            st_row, textvariable=self._step_type_var, state="readonly", width=20,
            values=list(STEP_HANDLERS.keys()),
        )
        step_type_combo.pack(side=tk.LEFT, padx=4)
        step_type_combo.bind("<<ComboboxSelected>>", self._on_step_type_changed)

        # Dynamic config area
        self._step_config_frame = ttk.Frame(config_frame)
        self._step_config_frame.pack(fill=tk.X, padx=4, pady=4)

        # Preview table
        ttk.Label(config_frame, text="Preview (5 rows):", style="Dim.TLabel").pack(anchor=tk.W, padx=4)
        self._prep_preview_tree = ttk.Treeview(config_frame, show="headings", height=5)
        self._prep_preview_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._prepare_steps: List[Dict[str, Any]] = []
        self._step_config_vars: Dict[str, tk.StringVar] = {}

    def _refresh_prep_combos(self) -> None:
        names = self.dataset_mgr.list_names()
        self._prep_ds_combo["values"] = names
        if names and not self._prep_ds_var.get():
            self._prep_ds_var.set(names[0])

    def _on_step_type_changed(self, _event: Any) -> None:
        for w in self._step_config_frame.winfo_children():
            w.destroy()
        self._step_config_vars.clear()

        step_type = self._step_type_var.get()
        ds_name = self._prep_ds_var.get()
        cols = []
        if ds_name and self.dataset_mgr.has(ds_name):
            cols = list(self.dataset_mgr.get_df(ds_name).columns)

        if step_type == "filter_rows":
            self._add_config_row("column", "Column:", combo_values=cols)
            self._add_config_row("operator", "Operator:", combo_values=["==", "!=", ">", ">=", "<", "<=", "contains", "not_contains"])
            self._add_config_row("value", "Value:")
        elif step_type == "drop_columns":
            self._add_config_row("columns", "Columns (comma sep):")
        elif step_type == "rename_column":
            self._add_config_row("old_name", "Old name:", combo_values=cols)
            self._add_config_row("new_name", "New name:")
        elif step_type == "fill_na":
            self._add_config_row("column", "Column:", combo_values=cols)
            self._add_config_row("method", "Method:", combo_values=["constant", "mean", "median", "mode", "ffill", "bfill"])
            self._add_config_row("value", "Value (if constant):")
        elif step_type in ("label_encode", "onehot_encode", "normalize", "standardize"):
            self._add_config_row("column", "Column:", combo_values=cols)
        elif step_type == "extract_datetime":
            self._add_config_row("column", "Column:", combo_values=cols)
            self._add_config_row("features", "Features (comma sep):", default="year,month,day,weekday")
        elif step_type == "custom_formula":
            self._add_config_row("new_column", "New column name:")
            self._add_config_row("formula", "Formula (pandas eval):")

    def _add_config_row(
        self, key: str, label: str,
        combo_values: Optional[List[str]] = None,
        default: str = "",
    ) -> None:
        row = ttk.Frame(self._step_config_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        if combo_values:
            w = ttk.Combobox(row, textvariable=var, values=combo_values, width=24)
        else:
            w = ttk.Entry(row, textvariable=var, width=26)
        w.pack(side=tk.LEFT, padx=4)
        self._step_config_vars[key] = var

    def _add_prepare_step(self) -> None:
        step_type = self._step_type_var.get()
        config: Dict[str, Any] = {}
        for k, v in self._step_config_vars.items():
            val = v.get()
            if k == "columns":
                config[k] = [c.strip() for c in val.split(",") if c.strip()]
            elif k == "features":
                config[k] = [c.strip() for c in val.split(",") if c.strip()]
            else:
                config[k] = val
        step = new_prepare_step(step_type, **config)
        self._prepare_steps.append(step)
        self._steps_listbox.insert(tk.END, f"{step_type} | {json.dumps(config, default=str)[:60]}")
        self._update_prepare_preview()

    def _remove_prepare_step(self) -> None:
        sel = self._steps_listbox.curselection()
        if sel:
            idx = sel[0]
            self._steps_listbox.delete(idx)
            self._prepare_steps.pop(idx)
            self._update_prepare_preview()

    def _update_prepare_preview(self) -> None:
        ds_name = self._prep_ds_var.get()
        if not ds_name or not self.dataset_mgr.has(ds_name):
            return
        try:
            df = self.dataset_mgr.get_df(ds_name)
            preview = RecipeEngine.preview_prepare(df, self._prepare_steps, n_rows=5)
            self._fill_preview_tree(self._prep_preview_tree, preview)
        except Exception as e:
            self._set_status(f"Preview error: {e}")

    def _fill_preview_tree(self, tree: ttk.Treeview, df: pd.DataFrame) -> None:
        tree.delete(*tree.get_children())
        cols = list(df.columns)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(70, min(150, len(c) * 9)))
        for _, row in df.iterrows():
            tree.insert("", tk.END, values=[str(v) for v in row.values])

    def _run_prepare_recipe(self) -> None:
        ds_name = self._prep_ds_var.get()
        if not ds_name:
            messagebox.showwarning("Prepare", "Select an input dataset.")
            return
        if not self._prepare_steps:
            messagebox.showwarning("Prepare", "Add at least one step.")
            return
        out_name = simpledialog.askstring("Output", "Output dataset name:", initialvalue=f"{ds_name}_prepared")
        if not out_name:
            return
        try:
            df = self.dataset_mgr.get_df(ds_name)
            result = RecipeEngine.execute_prepare(df, self._prepare_steps)
            self.dataset_mgr.register(result, out_name)
            self._refresh_tree()
            self._refresh_dt_combo()
            self._set_status(f"Prepare recipe → {out_name} ({len(result)} rows)")
            messagebox.showinfo("Done", f"Output dataset '{out_name}' created with {len(result)} rows.")
        except Exception as e:
            messagebox.showerror("Prepare Error", str(e))

    # -- notebook tab --------------------------------------------------------

    def _build_notebook_tab(self) -> None:
        toolbar = ttk.Frame(self._nb_frame)
        toolbar.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(toolbar, text="+ Cell", command=self._add_notebook_cell).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Run All", style="Accent.TButton", command=self._run_all_cells).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Outputs", command=self._clear_outputs).pack(side=tk.LEFT, padx=2)

        # Inject dataset selector
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(toolbar, text="Inject df:").pack(side=tk.LEFT)
        self._nb_ds_var = tk.StringVar()
        self._nb_ds_combo = ttk.Combobox(toolbar, textvariable=self._nb_ds_var, state="readonly", width=18)
        self._nb_ds_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Inject", command=self._inject_df).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="\u21bb", width=3, command=self._refresh_nb_datasets).pack(side=tk.LEFT, padx=2)

        # Scrollable cell area
        canvas = tk.Canvas(self._nb_frame, bg=Colors.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self._nb_frame, orient=tk.VERTICAL, command=canvas.yview)
        self._cells_frame = ttk.Frame(canvas)
        self._cells_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._cells_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mouse wheel
        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self._cell_widgets: Dict[str, Dict[str, Any]] = {}

        # Add an initial cell
        self._add_notebook_cell()

    def _refresh_nb_datasets(self) -> None:
        self._nb_ds_combo["values"] = self.dataset_mgr.list_names()

    def _inject_df(self) -> None:
        name = self._nb_ds_var.get()
        if name and self.dataset_mgr.has(name):
            self.notebook_mgr.inject("df", self.dataset_mgr.get_df(name))
            self._set_status(f"Injected '{name}' as `df` in notebook namespace")

    def _add_notebook_cell(self) -> None:
        cell = self.notebook_mgr.add_cell()
        self._render_cell_widget(cell)

    def _render_cell_widget(self, cell: NotebookCell) -> None:
        frame = ttk.Frame(self._cells_frame, style="Panel.TFrame")
        frame.pack(fill=tk.X, padx=8, pady=4)

        # Header
        header = ttk.Frame(frame, style="Panel.TFrame")
        header.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(header, text=f"In [{cell.execution_count}]:", style="Dim.TLabel").pack(side=tk.LEFT)
        ttk.Button(header, text="▶ Run", command=lambda c=cell: self._run_cell(c)).pack(side=tk.LEFT, padx=4)
        ttk.Button(header, text="✕", width=3, command=lambda c=cell: self._delete_cell(c)).pack(side=tk.RIGHT)
        ttk.Button(header, text="▼", width=3, command=lambda c=cell: self.notebook_mgr.move_cell(c.cell_id, 1) or self._rebuild_cells()).pack(side=tk.RIGHT, padx=1)
        ttk.Button(header, text="▲", width=3, command=lambda c=cell: self.notebook_mgr.move_cell(c.cell_id, -1) or self._rebuild_cells()).pack(side=tk.RIGHT, padx=1)

        # Code editor
        code_text = tk.Text(
            frame, height=5, wrap=tk.NONE,
            bg=Colors.INPUT_BG, fg=Colors.TEXT, font=FONT_CODE,
            insertbackground=Colors.TEXT, relief=tk.FLAT,
            selectbackground=Colors.ACCENT, selectforeground=Colors.TEXT_BRIGHT,
        )
        code_text.pack(fill=tk.X, padx=4, pady=2)
        code_text.insert("1.0", cell.source)

        # Output area
        output_text = tk.Text(
            frame, height=3, wrap=tk.WORD,
            bg="#1a1a2e", fg="#aaffaa", font=FONT_CODE,
            relief=tk.FLAT, state=tk.DISABLED,
        )
        output_text.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._cell_widgets[cell.cell_id] = {
            "frame": frame,
            "header": header,
            "code": code_text,
            "output": output_text,
            "cell": cell,
        }

    def _run_cell(self, cell: NotebookCell) -> None:
        widgets = self._cell_widgets.get(cell.cell_id)
        if not widgets:
            return
        code_text: tk.Text = widgets["code"]
        output_text: tk.Text = widgets["output"]
        cell.source = code_text.get("1.0", tk.END).rstrip()

        output = self.notebook_mgr.run_cell(cell.cell_id)

        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert("1.0", output)
        fg = "#ff6b6b" if cell.error else "#aaffaa"
        output_text.config(state=tk.DISABLED, fg=fg)

        # Update execution count label
        header: ttk.Frame = widgets["header"]
        for child in header.winfo_children():
            if isinstance(child, ttk.Label):
                child.config(text=f"In [{cell.execution_count}]:")
                break

    def _run_all_cells(self) -> None:
        for cell in self.notebook_mgr.cells:
            self._run_cell(cell)

    def _clear_outputs(self) -> None:
        for cid, widgets in self._cell_widgets.items():
            output_text: tk.Text = widgets["output"]
            output_text.config(state=tk.NORMAL)
            output_text.delete("1.0", tk.END)
            output_text.config(state=tk.DISABLED)

    def _delete_cell(self, cell: NotebookCell) -> None:
        self.notebook_mgr.delete_cell(cell.cell_id)
        w = self._cell_widgets.pop(cell.cell_id, None)
        if w:
            w["frame"].destroy()

    def _rebuild_cells(self) -> None:
        """Destroy and re-render all cell widgets in current order."""
        for cid, w in self._cell_widgets.items():
            # Save source from widget
            cell = w["cell"]
            cell.source = w["code"].get("1.0", tk.END).rstrip()
            w["frame"].destroy()
        self._cell_widgets.clear()
        for cell in self.notebook_mgr.cells:
            self._render_cell_widget(cell)

    # -- inspector -----------------------------------------------------------

    def _build_inspector(self) -> None:
        ttk.Label(self._inspector, text="Inspector", style="Heading.TLabel").pack(
            padx=8, pady=(8, 4), anchor=tk.W)
        self._insp_text = tk.Text(
            self._inspector, wrap=tk.WORD, height=30,
            bg=Colors.PANEL, fg=Colors.TEXT, font=FONT_SMALL,
            relief=tk.FLAT, state=tk.DISABLED,
            insertbackground=Colors.TEXT,
        )
        self._insp_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Schema editing buttons
        btn_frame = ttk.Frame(self._inspector)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_frame, text="Rename Col", command=self._rename_col_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Set Role", command=self._set_role_dialog).pack(side=tk.LEFT, padx=2)

    def _show_inspector_for_dataset(self, name: str) -> None:
        if not self.dataset_mgr.has(name):
            return
        info = self.dataset_mgr.get_info(name)
        df = self.dataset_mgr.get_df(name)
        lines = [
            f"Dataset: {name}",
            f"Rows: {info.row_count}  |  Cols: {info.col_count}",
            f"Memory: {info.memory}",
            "",
            "─── Columns ───",
        ]
        for c in info.columns:
            stats = column_stats(df, c["name"])
            role_tag = f" [{c.get('role', '')}]" if c.get("role") else ""
            lines.append(f"\n  {c['name']}{role_tag}")
            lines.append(f"    dtype: {stats['dtype']}")
            lines.append(f"    type:  {stats['semantic_type']}")
            lines.append(f"    nulls: {stats['null_count']}")
            lines.append(f"    unique: {stats['unique_count']}")
            if "mean" in stats and stats["mean"] is not None:
                lines.append(f"    min: {stats['min']:.4g}")
                lines.append(f"    max: {stats['max']:.4g}")
                lines.append(f"    mean: {stats['mean']:.4g}")

        self._insp_text.config(state=tk.NORMAL)
        self._insp_text.delete("1.0", tk.END)
        self._insp_text.insert("1.0", "\n".join(lines))
        self._insp_text.config(state=tk.DISABLED)

    def _rename_col_dialog(self) -> None:
        name = self._current_dataset
        if not name:
            return
        old = simpledialog.askstring("Rename", "Column to rename:")
        if not old:
            return
        new = simpledialog.askstring("Rename", f"New name for '{old}':")
        if not new:
            return
        try:
            self.dataset_mgr.rename_column(name, old, new)
            self._show_inspector_for_dataset(name)
            self._show_dataset_preview(name)
            self._set_status(f"Renamed '{old}' → '{new}'")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _set_role_dialog(self) -> None:
        name = self._current_dataset
        if not name:
            return
        col = simpledialog.askstring("Set Role", "Column name:")
        if not col:
            return
        role = simpledialog.askstring("Set Role", "Role (feature / target / ignore):", initialvalue="feature")
        if role:
            self.dataset_mgr.set_column_role(name, col, role)
            self._show_inspector_for_dataset(name)

    # -- status bar ----------------------------------------------------------

    def _build_statusbar(self) -> None:
        self._statusbar = ttk.Frame(self.root, style="Panel.TFrame")
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_label = ttk.Label(self._statusbar, text="Ready", style="Status.TLabel")
        self._status_label.pack(side=tk.LEFT, padx=8, pady=2)

        self._mem_label = ttk.Label(self._statusbar, text="", style="Status.TLabel")
        self._mem_label.pack(side=tk.RIGHT, padx=8, pady=2)

    def _set_status(self, msg: str) -> None:
        self._status_label.config(text=msg)

    # ======================================================================
    # DATA IMPORT / EXPORT
    # ======================================================================

    def _import_dataset(self) -> None:
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Import Dataset",
            filetypes=[("Data files", exts), ("All files", "*.*")],
        )
        if not path:
            return
        self._set_status(f"Importing {Path(path).name}…")

        def _load() -> str:
            info = self.dataset_mgr.import_file(Path(path))
            return info.name

        def _done(name: str) -> None:
            self._refresh_tree()
            self._refresh_dt_combo()
            self._current_dataset = name
            self._show_dataset_preview(name)
            self._show_inspector_for_dataset(name)
            self._set_status(f"Imported '{name}'")

        def _err(exc: Exception) -> None:
            messagebox.showerror("Import Error", str(exc))
            self._set_status("Import failed")

        run_in_background(_load, on_done=_done, on_error=_err)

    def _export_dataset(self) -> None:
        name = self._current_dataset
        if not name:
            messagebox.showinfo("Export", "Select a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Dataset",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"),
                       ("JSON", "*.json"), ("Parquet", "*.parquet")],
        )
        if path:
            try:
                self.dataset_mgr.export(name, Path(path))
                self._set_status(f"Exported '{name}' → {path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    # ======================================================================
    # PROJECT
    # ======================================================================

    def _new_project(self) -> None:
        name = simpledialog.askstring("New Project", "Project name:")
        if name:
            self._new_project_silent(name)

    def _new_project_silent(self, name: str) -> None:
        project = self.project_mgr.create_project(name)
        self.root.title(f"{self.TITLE} — {name}")
        self._set_status(f"Project '{name}' created")
        self._refresh_tree()

    def _open_project(self) -> None:
        path = filedialog.askdirectory(title="Open Project Folder")
        if not path:
            return
        try:
            project = self.project_mgr.open_project(Path(path))
            self.root.title(f"{self.TITLE} — {project.meta.name}")
            # Reload datasets
            data_dir = Path(path) / "data"
            if data_dir.exists():
                self.dataset_mgr.load_data_from_dir(data_dir, project.datasets)
            # Reload flow
            self._flow_canvas.load_flow_data(project.flow)
            # Reload notebook
            self.notebook_mgr.load_from_list(project.notebook_cells)
            self._rebuild_cells()
            self._refresh_tree()
            self._refresh_dt_combo()
            self._set_status(f"Opened project '{project.meta.name}'")
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    def _save_project(self) -> None:
        project = self.project_mgr.current_project
        if project is None:
            messagebox.showinfo("Save", "No project open.")
            return
        try:
            project.datasets = self.dataset_mgr.datasets_to_dict()
            project.flow = self._flow_canvas.get_flow_data()
            project.notebook_cells = self.notebook_mgr.to_list()
            data_dir = Path(project.meta.path) / "data"
            self.dataset_mgr.save_data_to_dir(data_dir)
            self.project_mgr.save_project(project)
            self._set_status(f"Project saved → {project.meta.path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ======================================================================
    # FLOW CANVAS CALLBACKS
    # ======================================================================

    def _on_flow_node_select(self, node: FlowNode) -> None:
        if node.node_type == "dataset" and self.dataset_mgr.has(node.label):
            self._current_dataset = node.label
            self._show_inspector_for_dataset(node.label)

    def _on_flow_run_node(self, node: FlowNode) -> None:
        if node.node_type == "recipe":
            config = node.config
            if not config:
                messagebox.showinfo("Flow", "Node has no recipe config. Double-click to configure.")
                return
            try:
                datasets = {n: self.dataset_mgr.get_df(n) for n in self.dataset_mgr.list_names()}
                result = RecipeEngine.execute_recipe(config.get("type", "prepare"), config, datasets)
                out_name = config.get("output", f"{node.label}_output")
                self.dataset_mgr.register(result, out_name)
                self._refresh_tree()
                self._set_status(f"Node '{node.label}' → {out_name} ({len(result)} rows)")
            except Exception as e:
                messagebox.showerror("Flow Error", str(e))

    def _on_flow_configure_node(self, node: FlowNode) -> None:
        """Open a config dialog for a flow node."""
        if node.node_type == "recipe":
            self._configure_recipe_node(node)
        elif node.node_type == "dataset":
            if self.dataset_mgr.has(node.label):
                self._current_dataset = node.label
                self._show_dataset_preview(node.label)
                self._show_inspector_for_dataset(node.label)
                self._notebook.select(1)  # Switch to Datasets tab

    def _configure_recipe_node(self, node: FlowNode) -> None:
        """Simple dialog to configure a recipe node."""
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Configure: {node.label}")
        dlg.geometry("450x350")
        dlg.configure(bg=Colors.PANEL)

        datasets = self.dataset_mgr.list_names()

        ttk.Label(dlg, text="Recipe Type:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        rtype_var = tk.StringVar(value=node.config.get("type", "prepare"))
        ttk.Combobox(dlg, textvariable=rtype_var, values=["prepare", "join", "groupby", "python"],
                     state="readonly", width=20).pack(padx=10, anchor=tk.W)

        ttk.Label(dlg, text="Input Dataset:").pack(anchor=tk.W, padx=10, pady=(8, 2))
        input_var = tk.StringVar(value=node.config.get("input", ""))
        ttk.Combobox(dlg, textvariable=input_var, values=datasets, state="readonly", width=24).pack(padx=10, anchor=tk.W)

        ttk.Label(dlg, text="Output Name:").pack(anchor=tk.W, padx=10, pady=(8, 2))
        output_var = tk.StringVar(value=node.config.get("output", f"{node.label}_output"))
        ttk.Entry(dlg, textvariable=output_var, width=26).pack(padx=10, anchor=tk.W)

        # For python recipe: code area
        ttk.Label(dlg, text="Code (Python recipe):").pack(anchor=tk.W, padx=10, pady=(8, 2))
        code_text = tk.Text(dlg, height=8, bg=Colors.INPUT_BG, fg=Colors.TEXT, font=FONT_CODE)
        code_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        code_text.insert("1.0", node.config.get("code", "output_df = df.copy()\n"))

        def _save() -> None:
            node.config = {
                "type": rtype_var.get(),
                "input": input_var.get(),
                "output": output_var.get(),
                "code": code_text.get("1.0", tk.END).rstrip(),
            }
            dlg.destroy()
            self._set_status(f"Configured node '{node.label}'")

        ttk.Button(dlg, text="Save", style="Accent.TButton", command=_save).pack(pady=8)

    def _run_pipeline(self) -> None:
        """Run all recipe nodes in topological order (simple: by edges)."""
        recipe_nodes = [n for n in self._flow_canvas.nodes.values() if n.node_type == "recipe"]
        if not recipe_nodes:
            self._set_status("No recipe nodes to run.")
            return
        count = 0
        for node in recipe_nodes:
            try:
                self._on_flow_run_node(node)
                count += 1
            except Exception as e:
                messagebox.showerror("Pipeline", f"Error on '{node.label}': {e}")
                break
        self._set_status(f"Pipeline complete – {count} nodes executed")

    # ======================================================================
    # TOOLS / MISC
    # ======================================================================

    def _reset_notebook_ns(self) -> None:
        self.notebook_mgr.reset_namespace()
        self._set_status("Notebook namespace reset")

    def _export_notebook(self) -> None:
        # Save current cell sources first
        for cid, w in self._cell_widgets.items():
            cell = w["cell"]
            cell.source = w["code"].get("1.0", tk.END).rstrip()
        script = self.notebook_mgr.export_as_py()
        path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python script", "*.py")],
        )
        if path:
            Path(path).write_text(script, encoding="utf-8")
            self._set_status(f"Notebook exported → {path}")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About DataikuLite DSS",
            "DataikuLite DSS v1.0\n\n"
            "A lightweight clone of Dataiku Data Science Studio\n"
            "built with Python, tkinter, pandas, and scikit-learn.\n\n"
            "© 2026 – Educational & Portfolio Project",
        )

    # ======================================================================
    # RUN
    # ======================================================================

    def run(self) -> None:
        """Start the tkinter main loop."""
        self._refresh_tree()
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch DataikuLite DSS."""
    app = DataikuLiteApp()
    app.run()


if __name__ == "__main__":
    main()
