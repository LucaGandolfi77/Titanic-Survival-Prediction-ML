"""
canvas.py – The Process Canvas: drag‑and‑drop operator graph with wires,
pan, zoom, selection, undo/redo, context menu, and auto‑layout.
"""
from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.operator_base import (
    Connection,
    Operator,
    PortType,
    get_operator_class,
    list_operator_types,
)
from engine.process_runner import Process
from gui.operator_node import (
    draw_operator_node,
    input_port_positions,
    output_port_positions,
)
from gui.theme import C, F, G
from gui.wire import draw_wire, draw_temp_wire

# ── Constants ──────────────────────────────────────────────────────────────

SNAP = G.GRID_SIZE


def _snap(v: float) -> float:
    return round(v / SNAP) * SNAP


# ═══════════════════════════════════════════════════════════════════════════
# ProcessCanvas
# ═══════════════════════════════════════════════════════════════════════════

class ProcessCanvas(ttk.Frame):
    """The central workspace canvas where operators are placed and connected.

    Public attributes / helpers the MainWindow uses:
      - ``process``: the current :class:`Process` object.
      - ``selected_op``: the currently selected :class:`Operator` (or None).
      - ``on_select_operator``: callback(op | None).
      - ``on_double_click_operator``: callback(op).
      - ``add_operator_at(op_type, x, y)``.
      - ``redraw()``.
      - ``auto_layout()``.
    """

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, style="Canvas.TFrame", **kw)
        self.process = Process()
        self.selected_op: Optional[Operator] = None
        self.on_select_operator: Optional[Callable] = None
        self.on_double_click_operator: Optional[Callable] = None

        # Undo / redo stacks (store serialised process snapshots)
        self._undo_stack: List[Dict] = []
        self._redo_stack: List[Dict] = []

        # Zoom / pan
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Interaction state
        self._drag_op: Optional[Operator] = None
        self._drag_start: Tuple[float, float] = (0, 0)
        self._wire_start: Optional[Tuple[str, str, PortType]] = None  # (op_id, port_name, type)
        self._wire_start_pos: Tuple[float, float] = (0, 0)
        self._panning: bool = False
        self._pan_start: Tuple[float, float] = (0, 0)

        # Build the canvas
        self.canvas = tk.Canvas(self, bg=C.CANVAS_BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True)

        # Toolbar
        self._build_toolbar()

        # Event bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_by(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_by(0.9))
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)

        # Keyboard
        self.canvas.bind("<Delete>", self._delete_selected)
        self.canvas.bind("<BackSpace>", self._delete_selected)
        self.canvas.bind("<Control-z>", self._undo)
        self.canvas.bind("<Control-y>", self._redo)
        self.canvas.bind("<Control-a>", self._select_all)
        self.canvas.focus_set()

    # ── toolbar ─────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, style="Panel.TFrame")
        bar.place(x=4, y=4)
        ttk.Button(bar, text="⊞ Auto‑layout", command=self.auto_layout,
                   style="TButton").pack(side="left", padx=2)
        ttk.Button(bar, text="⊕ Zoom In", command=lambda: self._zoom_by(1.2),
                   style="TButton").pack(side="left", padx=2)
        ttk.Button(bar, text="⊖ Zoom Out", command=lambda: self._zoom_by(0.8),
                   style="TButton").pack(side="left", padx=2)
        ttk.Button(bar, text="⊙ Reset", command=self._reset_view,
                   style="TButton").pack(side="left", padx=2)

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def add_operator_at(self, op_type: str, x: float = 200, y: float = 200) -> Operator:
        """Instantiate an operator and place it on the canvas."""
        self._push_undo()
        cls = get_operator_class(op_type)
        op = cls()
        op.x = _snap(x)
        op.y = _snap(y)
        self.process.add_operator(op)
        self.redraw()
        self._select(op)
        return op

    def delete_operator(self, op_id: str) -> None:
        self._push_undo()
        self.process.remove_operator(op_id)
        if self.selected_op and self.selected_op.op_id == op_id:
            self._select(None)
        self.redraw()

    def add_connection(self, from_id: str, from_port: str, to_id: str, to_port: str) -> None:
        self._push_undo()
        # Remove any existing connection to the same input port
        self.process.connections = [
            c for c in self.process.connections
            if not (c.to_op_id == to_id and c.to_port == to_port)
        ]
        conn = Connection(from_id, from_port, to_id, to_port)
        self.process.add_connection(conn)
        self.redraw()

    def load_process(self, proc: Process) -> None:
        """Replace the current process and redraw."""
        self._push_undo()
        self.process = proc
        self._select(None)
        self.redraw()

    def clear(self) -> None:
        self._push_undo()
        self.process = Process()
        self._select(None)
        self.redraw()

    # ── drawing ─────────────────────────────────────────────────────────

    def redraw(self) -> None:
        """Redraw all operators and wires."""
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_wires()
        for op in self.process.operators.values():
            selected = (self.selected_op is not None and op.op_id == self.selected_op.op_id)
            draw_operator_node(self.canvas, op, selected=selected)

    def _draw_grid(self) -> None:
        w = self.canvas.winfo_width() or 2000
        h = self.canvas.winfo_height() or 1400
        step = int(G.GRID_SIZE * self._zoom)
        if step < 5:
            return
        for x in range(0, w, step):
            self.canvas.create_line(x, 0, x, h, fill=C.BG_LIGHT, tags="grid")
        for y in range(0, h, step):
            self.canvas.create_line(0, y, w, y, fill=C.BG_LIGHT, tags="grid")

    def _draw_wires(self) -> None:
        for conn in self.process.connections:
            src = self.process.operators.get(conn.from_op_id)
            dst = self.process.operators.get(conn.to_op_id)
            if not src or not dst:
                continue
            # Find port positions
            sx, sy = self._port_center(src, conn.from_port, output=True)
            dx, dy = self._port_center(dst, conn.to_port, output=False)
            if sx is None:
                continue
            pt = src.outputs.get(conn.from_port)
            ptype = pt.port_type if pt else PortType.EXAMPLE_SET
            draw_wire(self.canvas, sx, sy, dx, dy, port_type=ptype, tag="wire")

    def _port_center(self, op: Operator, port_name: str, output: bool) -> Tuple:
        positions = output_port_positions(op) if output else input_port_positions(op)
        for name, px, py in positions:
            if name == port_name:
                return px, py
        return None, None

    # ── auto layout ─────────────────────────────────────────────────────

    def auto_layout(self) -> None:
        """Arrange operators left→right in topological order."""
        from engine.process_runner import topological_sort
        self._push_undo()
        try:
            order = topological_sort(self.process)
        except RuntimeError:
            order = list(self.process.operators.keys())

        # Assign columns based on longest path
        col_map: Dict[str, int] = {}
        adj: Dict[str, List[str]] = {}
        for c in self.process.connections:
            adj.setdefault(c.from_op_id, []).append(c.to_op_id)

        for oid in order:
            parents = [c.from_op_id for c in self.process.connections if c.to_op_id == oid]
            if not parents:
                col_map[oid] = 0
            else:
                col_map[oid] = max(col_map.get(p, 0) for p in parents) + 1

        # Group by column
        cols: Dict[int, List[str]] = {}
        for oid, col in col_map.items():
            cols.setdefault(col, []).append(oid)

        x_gap = G.NODE_W + 80
        y_gap = G.NODE_H + 40
        for col, op_ids in cols.items():
            for row, oid in enumerate(op_ids):
                op = self.process.operators[oid]
                op.x = 60 + col * x_gap
                op.y = 60 + row * y_gap
        self.redraw()

    # ══════════════════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════════════════

    def _on_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        x, y = event.x, event.y

        # Check if we clicked a port
        port_info = self._hit_port(x, y)
        if port_info:
            op_id, port_name, is_output, px, py, ptype = port_info
            if is_output:
                self._wire_start = (op_id, port_name, ptype)
                self._wire_start_pos = (px, py)
                return

        # Check if we clicked an operator body
        op = self._hit_operator(x, y)
        if op:
            self._select(op)
            self._drag_op = op
            self._drag_start = (x - op.x, y - op.y)
            return

        # Empty space → deselect + start pan
        self._select(None)
        self._panning = True
        self._pan_start = (x, y)

    def _on_drag(self, event: tk.Event) -> None:
        x, y = event.x, event.y

        # Dragging a wire
        if self._wire_start:
            self.canvas.delete("temp_wire")
            sx, sy = self._wire_start_pos
            ptype = self._wire_start[2]
            draw_temp_wire(self.canvas, sx, sy, x, y, port_type=ptype)
            return

        # Dragging an operator
        if self._drag_op:
            op = self._drag_op
            op.x = _snap(x - self._drag_start[0])
            op.y = _snap(y - self._drag_start[1])
            self.redraw()
            return

        # Panning
        if self._panning:
            dx = x - self._pan_start[0]
            dy = y - self._pan_start[1]
            self.canvas.scan_dragto(x, y, gain=1)
            return

    def _on_release(self, event: tk.Event) -> None:
        x, y = event.x, event.y

        # Completing a wire
        if self._wire_start:
            self.canvas.delete("temp_wire")
            port_info = self._hit_port(x, y)
            if port_info:
                to_op_id, to_port, is_output, _, _, _ = port_info
                if not is_output and to_op_id != self._wire_start[0]:
                    self.add_connection(
                        self._wire_start[0], self._wire_start[1],
                        to_op_id, to_port,
                    )
            self._wire_start = None
            self._wire_start_pos = (0, 0)
            return

        if self._drag_op:
            self._push_undo()
            self._drag_op = None

        self._panning = False

    def _on_double_click(self, event: tk.Event) -> None:
        op = self._hit_operator(event.x, event.y)
        if op and self.on_double_click_operator:
            self.on_double_click_operator(op)

    def _on_right_click(self, event: tk.Event) -> None:
        op = self._hit_operator(event.x, event.y)
        menu = tk.Menu(self.canvas, tearoff=0, bg=C.PANEL, fg=C.TEXT,
                       activebackground=C.SELECTED, activeforeground=C.TEXT,
                       font=F.NORMAL)
        if op:
            self._select(op)
            menu.add_command(label=f"Configure {op.display_name}",
                             command=lambda: self.on_double_click_operator and self.on_double_click_operator(op))
            menu.add_command(label="Delete", command=lambda: self.delete_operator(op.op_id))
            menu.add_separator()
        menu.add_command(label="Select All", command=lambda: self._select_all(None))
        menu.add_command(label="Auto Layout", command=self.auto_layout)
        menu.tk_popup(event.x_root, event.y_root)

    def _on_scroll(self, event: tk.Event) -> None:
        factor = 1.1 if event.delta > 0 else 0.9
        self._zoom_by(factor)

    # ── zoom ────────────────────────────────────────────────────────────

    def _zoom_by(self, factor: float) -> None:
        new = self._zoom * factor
        if G.MIN_ZOOM <= new <= G.MAX_ZOOM:
            self._zoom = new
            self.canvas.scale("all", 0, 0, factor, factor)

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self.redraw()

    # ── pan ─────────────────────────────────────────────────────────────

    def _start_pan(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ── hit testing ─────────────────────────────────────────────────────

    def _hit_operator(self, x: float, y: float) -> Optional[Operator]:
        for op in reversed(list(self.process.operators.values())):
            if op.x <= x <= op.x + G.NODE_W and op.y <= y <= op.y + G.NODE_H:
                return op
        return None

    def _hit_port(self, x: float, y: float) -> Optional[Tuple]:
        """Return (op_id, port_name, is_output, cx, cy, port_type) or None."""
        ps = G.PORT_SIZE // 2 + 4  # tolerance
        for op in self.process.operators.values():
            for name, px, py in output_port_positions(op):
                if abs(x - px) <= ps and abs(y - py) <= ps:
                    pt = op.outputs[name].port_type
                    return (op.op_id, name, True, px, py, pt)
            for name, px, py in input_port_positions(op):
                if abs(x - px) <= ps and abs(y - py) <= ps:
                    pt = op.inputs[name].port_type
                    return (op.op_id, name, False, px, py, pt)
        return None

    # ── selection ───────────────────────────────────────────────────────

    def _select(self, op: Optional[Operator]) -> None:
        self.selected_op = op
        if self.on_select_operator:
            self.on_select_operator(op)
        self.redraw()

    def _select_all(self, _event: Any) -> None:
        # Select first operator (full multi-select is beyond scope)
        ops = list(self.process.operators.values())
        if ops:
            self._select(ops[0])

    # ── delete ──────────────────────────────────────────────────────────

    def _delete_selected(self, _event: Any = None) -> None:
        if self.selected_op:
            self.delete_operator(self.selected_op.op_id)

    # ── undo / redo ─────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        snap = self.process.to_dict()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self, _event: Any = None) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.process.to_dict())
        data = self._undo_stack.pop()
        self.process = Process.from_dict(data)
        self._select(None)
        self.redraw()

    def _redo(self, _event: Any = None) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.process.to_dict())
        data = self._redo_stack.pop()
        self.process = Process.from_dict(data)
        self._select(None)
        self.redraw()
