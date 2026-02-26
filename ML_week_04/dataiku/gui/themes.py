"""
Dark theme constants and ttk style configuration for DataikuLite DSS.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

class Colors:
    """Centralised colour constants (dark theme)."""

    BG = "#1e1e2e"
    PANEL = "#2a2a3e"
    PANEL_LIGHT = "#353550"
    ACCENT = "#7c3aed"
    ACCENT_HOVER = "#9b5de5"
    TEXT = "#e0e0e0"
    TEXT_DIM = "#a0a0b0"
    TEXT_BRIGHT = "#ffffff"
    SUCCESS = "#22c55e"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    BORDER = "#3a3a50"
    SELECTION = "#4a3a6e"
    INPUT_BG = "#33334a"
    TREEVIEW_BG = "#252540"
    CANVAS_BG = "#1a1a30"
    NODE_DATASET = "#3b82f6"
    NODE_RECIPE = "#f59e0b"
    NODE_OUTPUT = "#22c55e"
    NODE_SELECTED = "#7c3aed"
    ARROW = "#888899"


# ---------------------------------------------------------------------------
# Font constants
# ---------------------------------------------------------------------------

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

# Fallbacks for Linux
import platform
if platform.system() == "Linux":
    FONT_FAMILY = "DejaVu Sans"
    FONT_MONO = "DejaVu Sans Mono"


FONT_NORMAL = (FONT_FAMILY, 10)
FONT_SMALL = (FONT_FAMILY, 9)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_HEADING = (FONT_FAMILY, 12, "bold")
FONT_TITLE = (FONT_FAMILY, 14, "bold")
FONT_CODE = (FONT_MONO, 10)
FONT_CODE_SMALL = (FONT_MONO, 9)


# ---------------------------------------------------------------------------
# Apply dark theme
# ---------------------------------------------------------------------------

def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    """Configure a dark ttk theme on *root* and return the Style object."""
    style = ttk.Style(root)

    # Use 'clam' as the base — it is the most customisable built-in theme
    style.theme_use("clam")

    # --- General widget styles ---
    style.configure(
        ".",
        background=Colors.BG,
        foreground=Colors.TEXT,
        fieldbackground=Colors.INPUT_BG,
        bordercolor=Colors.BORDER,
        darkcolor=Colors.PANEL,
        lightcolor=Colors.PANEL_LIGHT,
        troughcolor=Colors.PANEL,
        selectbackground=Colors.ACCENT,
        selectforeground=Colors.TEXT_BRIGHT,
        font=FONT_NORMAL,
    )

    # --- TFrame / TLabelframe ---
    style.configure("TFrame", background=Colors.BG)
    style.configure("Panel.TFrame", background=Colors.PANEL)
    style.configure("TLabelframe", background=Colors.PANEL, foreground=Colors.TEXT)
    style.configure("TLabelframe.Label", background=Colors.PANEL, foreground=Colors.TEXT, font=FONT_BOLD)

    # --- TLabel ---
    style.configure("TLabel", background=Colors.BG, foreground=Colors.TEXT, font=FONT_NORMAL)
    style.configure("Heading.TLabel", background=Colors.BG, foreground=Colors.TEXT_BRIGHT, font=FONT_HEADING)
    style.configure("Title.TLabel", background=Colors.BG, foreground=Colors.TEXT_BRIGHT, font=FONT_TITLE)
    style.configure("Dim.TLabel", background=Colors.BG, foreground=Colors.TEXT_DIM, font=FONT_SMALL)
    style.configure("Panel.TLabel", background=Colors.PANEL, foreground=Colors.TEXT)
    style.configure("Status.TLabel", background=Colors.PANEL, foreground=Colors.TEXT_DIM, font=FONT_SMALL)
    style.configure("Success.TLabel", background=Colors.BG, foreground=Colors.SUCCESS)
    style.configure("Error.TLabel", background=Colors.BG, foreground=Colors.ERROR)

    # --- TButton ---
    style.configure(
        "TButton",
        background=Colors.PANEL_LIGHT,
        foreground=Colors.TEXT,
        font=FONT_NORMAL,
        padding=(10, 4),
    )
    style.map(
        "TButton",
        background=[("active", Colors.ACCENT_HOVER), ("pressed", Colors.ACCENT)],
        foreground=[("active", Colors.TEXT_BRIGHT)],
    )
    style.configure(
        "Accent.TButton",
        background=Colors.ACCENT,
        foreground=Colors.TEXT_BRIGHT,
        font=FONT_BOLD,
        padding=(12, 6),
    )
    style.map(
        "Accent.TButton",
        background=[("active", Colors.ACCENT_HOVER), ("pressed", Colors.ACCENT)],
    )

    # --- TEntry ---
    style.configure(
        "TEntry",
        fieldbackground=Colors.INPUT_BG,
        foreground=Colors.TEXT,
        insertcolor=Colors.TEXT,
        padding=4,
    )

    # --- TCombobox ---
    style.configure(
        "TCombobox",
        fieldbackground=Colors.INPUT_BG,
        background=Colors.PANEL_LIGHT,
        foreground=Colors.TEXT,
        arrowcolor=Colors.TEXT,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", Colors.INPUT_BG)],
        selectbackground=[("readonly", Colors.ACCENT)],
    )

    # --- Treeview ---
    style.configure(
        "Treeview",
        background=Colors.TREEVIEW_BG,
        foreground=Colors.TEXT,
        fieldbackground=Colors.TREEVIEW_BG,
        font=FONT_NORMAL,
        rowheight=24,
    )
    style.configure(
        "Treeview.Heading",
        background=Colors.PANEL_LIGHT,
        foreground=Colors.TEXT_BRIGHT,
        font=FONT_BOLD,
    )
    style.map(
        "Treeview",
        background=[("selected", Colors.SELECTION)],
        foreground=[("selected", Colors.TEXT_BRIGHT)],
    )

    # --- TNotebook ---
    style.configure(
        "TNotebook",
        background=Colors.BG,
        bordercolor=Colors.BORDER,
    )
    style.configure(
        "TNotebook.Tab",
        background=Colors.PANEL,
        foreground=Colors.TEXT_DIM,
        padding=(12, 4),
        font=FONT_NORMAL,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", Colors.ACCENT)],
        foreground=[("selected", Colors.TEXT_BRIGHT)],
    )

    # --- Horizontal.TProgressbar ---
    style.configure(
        "Horizontal.TProgressbar",
        background=Colors.ACCENT,
        troughcolor=Colors.PANEL,
    )

    # --- TScale ---
    style.configure(
        "Horizontal.TScale",
        background=Colors.BG,
        troughcolor=Colors.PANEL,
    )

    # --- TPanedwindow ---
    style.configure("TPanedwindow", background=Colors.BORDER)

    # --- TScrollbar ---
    style.configure(
        "Vertical.TScrollbar",
        background=Colors.PANEL_LIGHT,
        troughcolor=Colors.PANEL,
        arrowcolor=Colors.TEXT_DIM,
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=Colors.PANEL_LIGHT,
        troughcolor=Colors.PANEL,
        arrowcolor=Colors.TEXT_DIM,
    )

    # --- TSeparator ---
    style.configure("TSeparator", background=Colors.BORDER)

    # --- TCheckbutton / TRadiobutton ---
    style.configure("TCheckbutton", background=Colors.BG, foreground=Colors.TEXT, font=FONT_NORMAL)
    style.configure("TRadiobutton", background=Colors.BG, foreground=Colors.TEXT, font=FONT_NORMAL)
    style.configure("Panel.TCheckbutton", background=Colors.PANEL, foreground=Colors.TEXT)
    style.configure("Panel.TRadiobutton", background=Colors.PANEL, foreground=Colors.TEXT)
    style.map("TCheckbutton", background=[("active", Colors.BG)])
    style.map("TRadiobutton", background=[("active", Colors.BG)])

    # --- TSpinbox ---
    style.configure(
        "TSpinbox",
        fieldbackground=Colors.INPUT_BG,
        foreground=Colors.TEXT,
        arrowcolor=Colors.TEXT,
        padding=4,
    )

    # Root window
    root.configure(bg=Colors.BG)
    root.option_add("*TCombobox*Listbox*Background", Colors.INPUT_BG)
    root.option_add("*TCombobox*Listbox*Foreground", Colors.TEXT)

    return style
