"""
theme.py – Colour palette, font constants, and ttk style configuration for
RapidMiner‑Lite's dark theme.
"""
from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk
from typing import Dict

# ── Colour palette ──────────────────────────────────────────────────────────

class C:
    """Named colour constants (dark‑theme palette inspired by RapidMiner)."""
    BG          = "#1e1e2e"
    BG_LIGHT    = "#252538"
    PANEL       = "#2a2a3e"
    PANEL_LIGHT = "#313148"
    CANVAS_BG   = "#20203a"

    ACCENT      = "#f59e0b"   # RapidMiner orange‑gold
    ACCENT_DARK = "#d97706"
    ACCENT2     = "#7c3aed"   # purple accent
    SUCCESS     = "#22c55e"
    ERROR       = "#ef4444"
    WARNING     = "#f59e0b"

    TEXT        = "#e0e0e0"
    TEXT_DIM    = "#9ca3af"
    TEXT_DARK   = "#6b7280"

    BORDER      = "#3a3a52"
    SELECTED    = "#3b3b5e"
    HOVER       = "#353550"

    # Port/wire colours
    WIRE_EXAMPLE = "#e8a838"
    WIRE_MODEL   = "#3a8ee6"
    WIRE_PERF    = "#4caf50"
    WIRE_ANY     = "#9e9e9e"

    # Category dot colours (for operator nodes)
    CAT_DATA    = "#42a5f5"
    CAT_TRANS   = "#66bb6a"
    CAT_FEAT    = "#ab47bc"
    CAT_MODEL   = "#ef5350"
    CAT_EVAL    = "#ffa726"
    CAT_VIZ     = "#26c6da"
    CAT_UTIL    = "#78909c"


# ── Fonts ───────────────────────────────────────────────────────────────────

_SYSTEM = platform.system()
_FONT_FAMILY = "Segoe UI" if _SYSTEM == "Windows" else (
    "SF Pro Text" if _SYSTEM == "Darwin" else "DejaVu Sans"
)

class F:
    """Font tuples used throughout the app."""
    NORMAL   = (_FONT_FAMILY, 10)
    SMALL    = (_FONT_FAMILY, 9)
    BOLD     = (_FONT_FAMILY, 10, "bold")
    HEADING  = (_FONT_FAMILY, 12, "bold")
    TITLE    = (_FONT_FAMILY, 14, "bold")
    MONO     = ("Consolas" if _SYSTEM == "Windows" else "DejaVu Sans Mono", 10)
    MONO_SM  = ("Consolas" if _SYSTEM == "Windows" else "DejaVu Sans Mono", 9)
    CANVAS   = (_FONT_FAMILY, 9)
    CANVAS_B = (_FONT_FAMILY, 9, "bold")


# ── Canvas geometry constants ──────────────────────────────────────────────

class G:
    """Geometry constants for the operator‑canvas."""
    NODE_W        = 150
    NODE_H        = 60
    NODE_RADIUS   = 10
    PORT_SIZE     = 10
    PORT_SPACING  = 20
    WIRE_CURVATURE = 0.5
    GRID_SIZE     = 20
    MIN_ZOOM      = 0.3
    MAX_ZOOM      = 3.0


# ── Apply theme ─────────────────────────────────────────────────────────────

def apply_theme(root: tk.Tk) -> ttk.Style:
    """Configure the ttk style for the entire application."""
    style = ttk.Style(root)
    style.theme_use("clam")

    # ── General ─────────────────────────────────────────────────────────
    style.configure(".", background=C.BG, foreground=C.TEXT,
                    fieldbackground=C.PANEL, font=F.NORMAL,
                    bordercolor=C.BORDER, darkcolor=C.BG,
                    lightcolor=C.BG_LIGHT,
                    troughcolor=C.BG_LIGHT, relief="flat")

    # ── Frames ──────────────────────────────────────────────────────────
    style.configure("TFrame", background=C.BG)
    style.configure("Panel.TFrame", background=C.PANEL)
    style.configure("Canvas.TFrame", background=C.CANVAS_BG)

    # ── Labels ──────────────────────────────────────────────────────────
    style.configure("TLabel", background=C.BG, foreground=C.TEXT)
    style.configure("Panel.TLabel", background=C.PANEL)
    style.configure("Heading.TLabel", font=F.HEADING)
    style.configure("Title.TLabel", font=F.TITLE, foreground=C.ACCENT)
    style.configure("Dim.TLabel", foreground=C.TEXT_DIM)
    style.configure("Status.TLabel", background=C.BG_LIGHT, foreground=C.TEXT_DIM,
                    font=F.SMALL, padding=(6, 2))

    # ── Buttons ─────────────────────────────────────────────────────────
    style.configure("TButton", background=C.PANEL_LIGHT, foreground=C.TEXT,
                    padding=(10, 4), font=F.NORMAL, borderwidth=0)
    style.map("TButton",
              background=[("active", C.HOVER), ("pressed", C.ACCENT_DARK)],
              foreground=[("active", C.TEXT)])

    style.configure("Accent.TButton", background=C.ACCENT, foreground="#000000",
                    font=F.BOLD, padding=(14, 5))
    style.map("Accent.TButton",
              background=[("active", C.ACCENT_DARK)])

    style.configure("Run.TButton", background=C.SUCCESS, foreground="#000000",
                    font=F.BOLD, padding=(14, 5))
    style.map("Run.TButton", background=[("active", "#16a34a")])

    style.configure("Stop.TButton", background=C.ERROR, foreground="#ffffff",
                    font=F.BOLD, padding=(14, 5))

    # ── Entry ───────────────────────────────────────────────────────────
    style.configure("TEntry", fieldbackground=C.PANEL_LIGHT,
                    foreground=C.TEXT, insertcolor=C.TEXT,
                    borderwidth=1, padding=(4, 3))

    # ── Combobox ────────────────────────────────────────────────────────
    style.configure("TCombobox", fieldbackground=C.PANEL_LIGHT,
                    foreground=C.TEXT, selectbackground=C.SELECTED,
                    selectforeground=C.TEXT, padding=(4, 3))
    style.map("TCombobox", fieldbackground=[("readonly", C.PANEL_LIGHT)])

    # ── Spinbox ─────────────────────────────────────────────────────────
    style.configure("TSpinbox", fieldbackground=C.PANEL_LIGHT,
                    foreground=C.TEXT, arrowcolor=C.TEXT)

    # ── Checkbutton ─────────────────────────────────────────────────────
    style.configure("TCheckbutton", background=C.BG, foreground=C.TEXT)
    style.configure("Panel.TCheckbutton", background=C.PANEL)

    # ── Notebook (tabs) ─────────────────────────────────────────────────
    style.configure("TNotebook", background=C.BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=C.PANEL, foreground=C.TEXT_DIM,
                    padding=(12, 4), font=F.NORMAL)
    style.map("TNotebook.Tab",
              background=[("selected", C.BG_LIGHT)],
              foreground=[("selected", C.TEXT)])

    # ── Treeview ────────────────────────────────────────────────────────
    style.configure("Treeview", background=C.PANEL, foreground=C.TEXT,
                    fieldbackground=C.PANEL, rowheight=24, font=F.SMALL)
    style.configure("Treeview.Heading", background=C.BG_LIGHT,
                    foreground=C.TEXT, font=F.BOLD)
    style.map("Treeview",
              background=[("selected", C.SELECTED)],
              foreground=[("selected", C.TEXT)])

    # ── Scrollbar ───────────────────────────────────────────────────────
    style.configure("Vertical.TScrollbar", background=C.PANEL,
                    troughcolor=C.BG, arrowcolor=C.TEXT_DIM,
                    borderwidth=0, width=10)
    style.configure("Horizontal.TScrollbar", background=C.PANEL,
                    troughcolor=C.BG, arrowcolor=C.TEXT_DIM,
                    borderwidth=0)

    # ── Progressbar ─────────────────────────────────────────────────────
    style.configure("TProgressbar", troughcolor=C.BG_LIGHT,
                    background=C.ACCENT, borderwidth=0)
    style.configure("Green.Horizontal.TProgressbar",
                    troughcolor=C.BG_LIGHT, background=C.SUCCESS)

    # ── Panedwindow ─────────────────────────────────────────────────────
    style.configure("TPanedwindow", background=C.BORDER)
    style.configure("Sash", sashthickness=4, handlesize=8)

    # ── Separator ───────────────────────────────────────────────────────
    style.configure("TSeparator", background=C.BORDER)

    # ── LabelFrame ──────────────────────────────────────────────────────
    style.configure("TLabelframe", background=C.PANEL, foreground=C.TEXT,
                    bordercolor=C.BORDER)
    style.configure("TLabelframe.Label", background=C.PANEL, foreground=C.ACCENT,
                    font=F.BOLD)

    return style
