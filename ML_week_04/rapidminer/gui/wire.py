"""
wire.py – Bézier wire drawing between operator ports on the canvas.
"""
from __future__ import annotations

import math
import tkinter as tk
from typing import Tuple

from engine.operator_base import PortType, PORT_COLOURS
from gui.theme import C, G

# ── Colour by port type ────────────────────────────────────────────────────

def wire_colour(port_type: PortType) -> str:
    return PORT_COLOURS.get(port_type, C.WIRE_ANY)


INVALID_WIRE_COLOUR = C.ERROR

# ── Bézier helpers ─────────────────────────────────────────────────────────

def _bezier_points(
    x0: float, y0: float, x1: float, y1: float,
    curvature: float = G.WIRE_CURVATURE, segments: int = 32,
) -> list[float]:
    """Return a flat list of (x, y, …) coordinates for a cubic Bézier."""
    dx = abs(x1 - x0) * curvature
    # Control points: handles push right from source, left from target
    cx0, cy0 = x0 + dx, y0
    cx1, cy1 = x1 - dx, y1

    pts: list[float] = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        x = (u**3 * x0 + 3 * u**2 * t * cx0
             + 3 * u * t**2 * cx1 + t**3 * x1)
        y = (u**3 * y0 + 3 * u**2 * t * cy0
             + 3 * u * t**2 * cy1 + t**3 * y1)
        pts.extend((x, y))
    return pts


# ── Public draw function ───────────────────────────────────────────────────

def draw_wire(
    canvas: tk.Canvas,
    x0: float, y0: float,
    x1: float, y1: float,
    port_type: PortType = PortType.EXAMPLE_SET,
    invalid: bool = False,
    tag: str = "wire",
    width: float = 2.5,
    dash: Tuple[int, ...] | None = None,
) -> int:
    """Draw a Bézier wire on *canvas* and return the canvas item id.

    Args:
        canvas: the tkinter Canvas widget.
        x0, y0: source point (output port centre).
        x1, y1: target point (input port centre).
        port_type: determines colour.
        invalid: if True, draw in red.
        tag: canvas tag string.
        width: line width.
        dash: optional dash pattern (e.g. (6, 4)).

    Returns:
        The canvas item id of the line.
    """
    colour = INVALID_WIRE_COLOUR if invalid else wire_colour(port_type)
    pts = _bezier_points(x0, y0, x1, y1)
    kwargs = dict(
        fill=colour, width=width, smooth=True, tags=tag,
        capstyle=tk.ROUND, joinstyle=tk.ROUND,
    )
    if dash:
        kwargs["dash"] = dash
    item = canvas.create_line(*pts, **kwargs)
    return item


def draw_temp_wire(
    canvas: tk.Canvas,
    x0: float, y0: float,
    x1: float, y1: float,
    port_type: PortType = PortType.EXAMPLE_SET,
) -> int:
    """Draw a dashed temporary wire while the user is dragging."""
    return draw_wire(canvas, x0, y0, x1, y1,
                     port_type=port_type, tag="temp_wire",
                     width=2.0, dash=(6, 4))
