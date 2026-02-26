"""
operator_node.py – Draw a single operator as a rounded rectangle on the canvas
with named input/output ports and a category‑coloured dot.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, Dict, List, Tuple

from engine.operator_base import (
    CATEGORY_COLOURS,
    Operator,
    OpCategory,
    Port,
    PortType,
)
from gui.theme import C, F, G

# ── Port position helpers ──────────────────────────────────────────────────

def input_port_positions(op: Operator) -> List[Tuple[str, float, float]]:
    """Return [(port_name, cx, cy), …] for each input port."""
    ports = list(op.inputs.keys())
    n = len(ports)
    if n == 0:
        return []
    spacing = min(G.PORT_SPACING, (G.NODE_H - 10) / max(n, 1))
    total = spacing * (n - 1)
    start_y = op.y + G.NODE_H / 2 - total / 2
    return [(name, op.x, start_y + i * spacing) for i, name in enumerate(ports)]


def output_port_positions(op: Operator) -> List[Tuple[str, float, float]]:
    """Return [(port_name, cx, cy), …] for each output port."""
    ports = list(op.outputs.keys())
    n = len(ports)
    if n == 0:
        return []
    spacing = min(G.PORT_SPACING, (G.NODE_H - 10) / max(n, 1))
    total = spacing * (n - 1)
    start_y = op.y + G.NODE_H / 2 - total / 2
    return [(name, op.x + G.NODE_W, start_y + i * spacing) for i, name in enumerate(ports)]


# ── Rounded rectangle helper ──────────────────────────────────────────────

def _round_rect(
    canvas: tk.Canvas, x: float, y: float, w: float, h: float,
    r: float, **kwargs: Any,
) -> int:
    """Draw a rounded rectangle and return the canvas item id."""
    pts = [
        x + r, y,
        x + w - r, y,
        x + w, y, x + w, y + r,
        x + w, y + h - r,
        x + w, y + h, x + w - r, y + h,
        x + r, y + h,
        x, y + h, x, y + h - r,
        x, y + r,
        x, y, x + r, y,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


# ── Draw one operator node ─────────────────────────────────────────────────

def draw_operator_node(
    canvas: tk.Canvas,
    op: Operator,
    selected: bool = False,
) -> str:
    """Draw an operator node on *canvas*.  Returns the tag prefix ``op_<id>``."""
    tag = f"op_{op.op_id}"
    canvas.delete(tag)

    x, y = op.x, op.y
    w, h = G.NODE_W, G.NODE_H
    r = G.NODE_RADIUS

    # ── body ────────────────────────────────────────────────────────────
    outline = C.ACCENT if selected else C.BORDER
    fill = C.PANEL_LIGHT if selected else C.PANEL

    # Execution state overrides
    if op.error:
        outline = C.ERROR
    elif op.executed:
        outline = C.SUCCESS

    _round_rect(canvas, x, y, w, h, r, fill=fill, outline=outline,
                width=2, tags=(tag, "node_body"))

    # ── category dot ────────────────────────────────────────────────────
    cat_col = CATEGORY_COLOURS.get(op.category, C.CAT_UTIL)
    dot_r = 5
    canvas.create_oval(x + 8, y + 8, x + 8 + dot_r * 2, y + 8 + dot_r * 2,
                       fill=cat_col, outline="", tags=(tag,))

    # ── label ───────────────────────────────────────────────────────────
    canvas.create_text(x + w / 2, y + h / 2,
                       text=op.display_name, fill=C.TEXT,
                       font=F.CANVAS_B, tags=(tag, "node_label"),
                       width=w - 20)

    # ── input ports ─────────────────────────────────────────────────────
    ps = G.PORT_SIZE // 2
    for name, px, py in input_port_positions(op):
        pt = op.inputs[name].port_type
        col = _port_colour(pt)
        pid = f"{tag}_inp_{name}"
        canvas.create_rectangle(px - ps, py - ps, px + ps, py + ps,
                                fill=col, outline=C.BORDER, width=1,
                                tags=(tag, pid, "port", "input_port"))
        canvas.create_text(px + ps + 3, py, text=name[:3], anchor="w",
                           fill=C.TEXT_DIM, font=F.CANVAS, tags=(tag,))

    # ── output ports ────────────────────────────────────────────────────
    for name, px, py in output_port_positions(op):
        pt = op.outputs[name].port_type
        col = _port_colour(pt)
        pid = f"{tag}_out_{name}"
        canvas.create_rectangle(px - ps, py - ps, px + ps, py + ps,
                                fill=col, outline=C.BORDER, width=1,
                                tags=(tag, pid, "port", "output_port"))
        canvas.create_text(px - ps - 3, py, text=name[:3], anchor="e",
                           fill=C.TEXT_DIM, font=F.CANVAS, tags=(tag,))

    return tag


def _port_colour(pt: PortType) -> str:
    return {
        PortType.EXAMPLE_SET: C.WIRE_EXAMPLE,
        PortType.MODEL:       C.WIRE_MODEL,
        PortType.PERFORMANCE: C.WIRE_PERF,
        PortType.ANY:         C.WIRE_ANY,
    }.get(pt, C.WIRE_ANY)
