#!/usr/bin/env python3
"""
main.py – Entry point for RapidMiner‑Lite.

Builds the MainWindow (3‑panel layout + menu bar + status bar),
wires up all GUI panels, and starts the tkinter event loop.
"""
from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

# ── Ensure the package root is on sys.path ──────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── Engine imports ──────────────────────────────────────────────────────
from engine.operator_base import (
    Operator,
    Connection,
    get_operator_class,
    list_operator_types,
    operators_by_category,
)
from engine.process_runner import Process, ProcessRunner
from engine import repository as repo

# Force‑register all operator modules so the palette discovers them
import engine.operators_data       # noqa: F401
import engine.operators_transform  # noqa: F401
import engine.operators_feature    # noqa: F401
import engine.operators_model      # noqa: F401
import engine.operators_eval       # noqa: F401
import engine.operators_viz        # noqa: F401

# ── GUI imports ─────────────────────────────────────────────────────────
from gui.theme import C, F, apply_theme
from gui.canvas import ProcessCanvas
from gui.left_panel import LeftPanel
from gui.right_panel import RightPanel
from gui.results_panel import ResultsPanel
from gui.automodel_wizard import AutoModelWizard

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("rapidminer")


# ═══════════════════════════════════════════════════════════════════════════
# MainWindow
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow:
    """Top‑level application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RapidMiner Lite — Visual Data Science")
        self.root.geometry("1440x900")
        self.root.minsize(960, 600)
        self.root.configure(bg=C.BG)

        self.style = apply_theme(self.root)
        self.runner = ProcessRunner()
        self._runner_thread: Optional[threading.Thread] = None

        self._build_menu()
        self._build_layout()
        self._wire_callbacks()
        self._build_statusbar()

        # Load sample data hints
        repo._ensure_dirs()

    # ── Menu bar ────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_cfg = dict(bg=C.PANEL, fg=C.TEXT, activebackground=C.SELECTED,
                        activeforeground=C.TEXT, font=F.NORMAL, tearoff=0)
        mb = tk.Menu(self.root, **menu_cfg)

        # File
        fm = tk.Menu(mb, **menu_cfg)
        fm.add_command(label="New Process", accelerator="Ctrl+N",
                       command=self._new_process)
        fm.add_command(label="Open Process…", accelerator="Ctrl+O",
                       command=self._open_process)
        fm.add_command(label="Save Process", accelerator="Ctrl+S",
                       command=self._save_process)
        fm.add_command(label="Save As…", accelerator="Ctrl+Shift+S",
                       command=self._save_process_as)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.quit)
        mb.add_cascade(label="File", menu=fm)

        # Edit
        em = tk.Menu(mb, **menu_cfg)
        em.add_command(label="Undo", accelerator="Ctrl+Z",
                       command=lambda: self.canvas._undo())
        em.add_command(label="Redo", accelerator="Ctrl+Y",
                       command=lambda: self.canvas._redo())
        em.add_separator()
        em.add_command(label="Delete Selected", accelerator="Del",
                       command=lambda: self.canvas._delete_selected())
        mb.add_cascade(label="Edit", menu=em)

        # Process
        pm = tk.Menu(mb, **menu_cfg)
        pm.add_command(label="▶ Run Process", accelerator="F5",
                       command=self._run_process)
        pm.add_command(label="■ Stop", command=self._stop_process)
        pm.add_separator()
        pm.add_command(label="Auto Layout", command=self.canvas_auto_layout)
        mb.add_cascade(label="Process", menu=pm)

        # Tools
        tm = tk.Menu(mb, **menu_cfg)
        tm.add_command(label="✨ AutoModel Wizard…",
                       command=self._open_automodel)
        mb.add_cascade(label="Tools", menu=tm)

        # Help
        hm = tk.Menu(mb, **menu_cfg)
        hm.add_command(label="About", command=self._about)
        mb.add_cascade(label="Help", menu=hm)

        self.root.config(menu=mb)

        # Keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self._new_process())
        self.root.bind("<Control-o>", lambda e: self._open_process())
        self.root.bind("<Control-s>", lambda e: self._save_process())
        self.root.bind("<F5>", lambda e: self._run_process())

    # ── Layout ──────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Horizontal paned window: left | centre | right
        self._hpw = ttk.PanedWindow(self.root, orient="horizontal")
        self._hpw.pack(fill="both", expand=True)

        # Left panel (repo + palette)
        self.left_panel = LeftPanel(self._hpw)
        self._hpw.add(self.left_panel, weight=0)

        # Centre: vertical paned (canvas on top, results on bottom)
        self._vpw = ttk.PanedWindow(self._hpw, orient="vertical")
        self._hpw.add(self._vpw, weight=3)

        self.canvas = ProcessCanvas(self._vpw)
        self._vpw.add(self.canvas, weight=3)

        self.results = ResultsPanel(self._vpw)
        self._vpw.add(self.results, weight=1)

        # Right panel (params)
        self.right_panel = RightPanel(self._hpw)
        self._hpw.add(self.right_panel, weight=0)

    # ── Status bar ──────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, style="TFrame")
        bar.pack(fill="x", side="bottom")
        self._status = ttk.Label(bar, text="Ready", style="Status.TLabel")
        self._status.pack(fill="x")

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    # ── Callbacks ───────────────────────────────────────────────────────

    def _wire_callbacks(self) -> None:
        # Canvas → right panel
        self.canvas.on_select_operator = self.right_panel.show_operator
        self.canvas.on_double_click_operator = self.right_panel.show_operator

        # Right panel → canvas redraw
        self.right_panel.on_params_changed = lambda op: self.canvas.redraw()

        # Left panel – add operator
        self.left_panel.palette.on_add_operator = self._add_operator_from_palette

        # Left panel – open process
        self.left_panel.repository.on_open_process = self._load_process_file

        # Runner callbacks
        self.runner.on_start = lambda total: self.root.after(
            0, lambda: self._set_status(f"Running… (0/{total})"))
        self.runner.on_operator_done = lambda oid, name, t: self.root.after(
            0, lambda: self._set_status(f"✓ {name} ({t:.2f}s)"))
        self.runner.on_operator_error = lambda oid, name, e: self.root.after(
            0, lambda: self.results.log(f"✗ {name}: {e}", "error"))
        self.runner.on_log = lambda msg: self.root.after(
            0, lambda: self.results.log(msg))
        self.runner.on_complete = lambda t, res: self.root.after(
            0, lambda: self._on_run_complete(t, res))

    def _add_operator_from_palette(self, op_type: str) -> None:
        self.canvas.add_operator_at(op_type, 250, 200)

    # ── File menu actions ───────────────────────────────────────────────

    def _new_process(self) -> None:
        self.canvas.clear()
        self._set_status("New process created.")

    def _open_process(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("RapidMiner Process", "*.rmp"), ("All", "*.*")])
        if path:
            self._load_process_file(Path(path))

    def _load_process_file(self, path: Path) -> None:
        try:
            proc = repo.load_process(path)
            self.canvas.load_process(proc)
            self._set_status(f"Loaded: {path.name}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _save_process(self) -> None:
        proc = self.canvas.process
        path = repo.save_process(proc)
        self._set_status(f"Saved: {path.name}")
        self.left_panel.repository.refresh()

    def _save_process_as(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".rmp",
            filetypes=[("RapidMiner Process", "*.rmp")])
        if path:
            repo.save_process(self.canvas.process, Path(path))
            self._set_status(f"Saved: {Path(path).name}")

    # ── Run / stop ──────────────────────────────────────────────────────

    def _run_process(self) -> None:
        proc = self.canvas.process
        if not proc.operators:
            self._set_status("Nothing to run — add operators first.")
            return
        self.results.log_view.clear()
        self.results.notebook.select(4)  # Log tab
        self._set_status("Running…")
        self._runner_thread = self.runner.run_async(proc)

    def _stop_process(self) -> None:
        self.runner.stop()
        self._set_status("Stopped.")

    def _on_run_complete(self, total_time: float, results: Dict) -> None:
        self._set_status(f"✓ Completed in {total_time:.2f}s")
        self.canvas.redraw()
        self.results.show_results(results)

    # ── Canvas convenience ──────────────────────────────────────────────

    def canvas_auto_layout(self) -> None:
        self.canvas.auto_layout()

    # ── AutoModel ───────────────────────────────────────────────────────

    def _open_automodel(self) -> None:
        wiz = AutoModelWizard(self.root)
        wiz.on_complete = lambda: None  # future hook

    # ── About ───────────────────────────────────────────────────────────

    def _about(self) -> None:
        messagebox.showinfo(
            "About RapidMiner Lite",
            "RapidMiner Lite\n"
            "A visual data‑science pipeline builder.\n\n"
            "Built with Python, tkinter, scikit‑learn & pandas.\n"
            "© 2025 — Educational project.")

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
