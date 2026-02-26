"""
FlowCanvas – drag-and-drop pipeline canvas built on tkinter Canvas.

Implements:
  • Dataset nodes, Recipe nodes, Output nodes
  • Drag to reposition
  • Right-click context menu for adding / deleting / configuring nodes
  • Arrow connections between nodes
  • Run individual node or full pipeline
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Any, Callable, Dict, List, Optional, Tuple

from gui.themes import Colors, FONT_NORMAL, FONT_BOLD, FONT_SMALL

import uuid


# ---------------------------------------------------------------------------
# Data structs
# ---------------------------------------------------------------------------

class FlowNode:
    """A single node on the canvas."""

    def __init__(
        self,
        node_id: str,
        label: str,
        node_type: str,
        x: float,
        y: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_id = node_id
        self.label = label
        self.node_type = node_type  # 'dataset', 'recipe', 'output'
        self.x = x
        self.y = y
        self.config: Dict[str, Any] = config or {}
        # Canvas item ids
        self.rect_id: Optional[int] = None
        self.text_id: Optional[int] = None
        self.icon_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "x": self.x,
            "y": self.y,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowNode":
        return cls(
            node_id=d["node_id"],
            label=d["label"],
            node_type=d["node_type"],
            x=d["x"],
            y=d["y"],
            config=d.get("config", {}),
        )


class FlowEdge:
    """A directed edge between two nodes."""

    def __init__(self, source_id: str, target_id: str) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.line_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowEdge":
        return cls(source_id=d["source_id"], target_id=d["target_id"])


# ---------------------------------------------------------------------------
# Node dimensions
# ---------------------------------------------------------------------------

NODE_W = 140
NODE_H = 50
NODE_COLORS = {
    "dataset": Colors.NODE_DATASET,
    "recipe": Colors.NODE_RECIPE,
    "output": Colors.NODE_OUTPUT,
}
NODE_ICONS = {
    "dataset": "\u2b58",    # ● circle
    "recipe": "\u2699",     # ⚙ gear
    "output": "\u2714",     # ✔ check
}


# ---------------------------------------------------------------------------
# FlowCanvas widget
# ---------------------------------------------------------------------------

class FlowCanvas(ttk.Frame):
    """Interactive pipeline canvas."""

    def __init__(
        self,
        parent: tk.Widget,
        on_node_select: Optional[Callable[[FlowNode], None]] = None,
        on_run_node: Optional[Callable[[FlowNode], None]] = None,
        on_run_all: Optional[Callable[[], None]] = None,
        on_configure_node: Optional[Callable[[FlowNode], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.on_node_select = on_node_select
        self.on_run_node = on_run_node
        self.on_run_all = on_run_all
        self.on_configure_node = on_configure_node

        self.nodes: Dict[str, FlowNode] = {}
        self.edges: List[FlowEdge] = []
        self._selected_node: Optional[str] = None
        self._drag_data: Dict[str, Any] = {}
        self._connect_mode: bool = False
        self._connect_source: Optional[str] = None

        self._build_ui()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=2, pady=2)

        ttk.Button(toolbar, text="+ Dataset", command=self._add_dataset_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="+ Recipe", command=self._add_recipe_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="+ Output", command=self._add_output_node).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="Connect", command=self._toggle_connect_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Delete", command=self._delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="\u25b6 Run All", style="Accent.TButton", command=self._run_all).pack(side=tk.LEFT, padx=2)

        self._connect_label = ttk.Label(toolbar, text="", style="Dim.TLabel")
        self._connect_label.pack(side=tk.LEFT, padx=8)

        # Canvas
        self.canvas = tk.Canvas(
            self,
            bg=Colors.CANVAS_BG,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

    # -- adding nodes --------------------------------------------------------

    def _add_dataset_node(self) -> None:
        name = simpledialog.askstring("Dataset Node", "Dataset name:")
        if name:
            self.add_node(name, "dataset")

    def _add_recipe_node(self) -> None:
        name = simpledialog.askstring("Recipe Node", "Recipe name:")
        if name:
            self.add_node(name, "recipe")

    def _add_output_node(self) -> None:
        name = simpledialog.askstring("Output Node", "Output dataset name:")
        if name:
            self.add_node(name, "output")

    def add_node(
        self,
        label: str,
        node_type: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> FlowNode:
        """Create and draw a new node on the canvas."""
        if x is None:
            x = 80 + len(self.nodes) * 180
        if y is None:
            y = 100
        nid = node_id or uuid.uuid4().hex[:10]
        node = FlowNode(nid, label, node_type, x, y, config)
        self.nodes[nid] = node
        self._draw_node(node)
        return node

    # -- drawing -------------------------------------------------------------

    def _draw_node(self, node: FlowNode) -> None:
        x, y = node.x, node.y
        color = NODE_COLORS.get(node.node_type, Colors.PANEL_LIGHT)
        icon = NODE_ICONS.get(node.node_type, "")

        rect = self.canvas.create_rectangle(
            x, y, x + NODE_W, y + NODE_H,
            fill=color, outline=Colors.BORDER, width=2,
            tags=("node", node.node_id),
        )
        text = self.canvas.create_text(
            x + NODE_W / 2, y + NODE_H / 2 + 2,
            text=node.label, fill="white", font=FONT_BOLD,
            tags=("node", node.node_id),
        )
        icon_item = self.canvas.create_text(
            x + 14, y + 14,
            text=icon, fill="white", font=FONT_SMALL,
            tags=("node", node.node_id),
        )
        node.rect_id = rect
        node.text_id = text
        node.icon_id = icon_item

        # Bind drag
        for item in (rect, text, icon_item):
            self.canvas.tag_bind(item, "<ButtonPress-1>", lambda e, n=node: self._on_node_press(e, n))
            self.canvas.tag_bind(item, "<B1-Motion>", lambda e, n=node: self._on_node_drag(e, n))
            self.canvas.tag_bind(item, "<ButtonRelease-1>", lambda e, n=node: self._on_node_release(e, n))
            self.canvas.tag_bind(item, "<Double-Button-1>", lambda e, n=node: self._on_node_double_click(e, n))

    def _redraw_edges(self) -> None:
        """Redraw all edge arrows."""
        for edge in self.edges:
            if edge.line_id is not None:
                self.canvas.delete(edge.line_id)
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src and tgt:
                x1 = src.x + NODE_W
                y1 = src.y + NODE_H / 2
                x2 = tgt.x
                y2 = tgt.y + NODE_H / 2
                edge.line_id = self.canvas.create_line(
                    x1, y1, x2, y2,
                    arrow=tk.LAST, fill=Colors.ARROW, width=2,
                    tags="edge",
                )
                self.canvas.tag_lower("edge")

    # -- interaction ---------------------------------------------------------

    def _on_node_press(self, event: tk.Event, node: FlowNode) -> None:  # type: ignore[type-arg]
        if self._connect_mode:
            self._handle_connect_click(node)
            return
        self._drag_data = {"x": event.x, "y": event.y}
        self._select_node(node.node_id)

    def _on_node_drag(self, event: tk.Event, node: FlowNode) -> None:  # type: ignore[type-arg]
        if self._connect_mode:
            return
        dx = event.x - self._drag_data.get("x", event.x)
        dy = event.y - self._drag_data.get("y", event.y)
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        for item in (node.rect_id, node.text_id, node.icon_id):
            if item is not None:
                self.canvas.move(item, dx, dy)
        node.x += dx
        node.y += dy
        self._redraw_edges()

    def _on_node_release(self, event: tk.Event, node: FlowNode) -> None:  # type: ignore[type-arg]
        pass

    def _on_node_double_click(self, event: tk.Event, node: FlowNode) -> None:  # type: ignore[type-arg]
        if self.on_configure_node:
            self.on_configure_node(node)

    def _on_canvas_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        # Deselect when clicking empty space
        items = self.canvas.find_closest(event.x, event.y)
        if not items:
            self._deselect()

    def _on_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        menu = tk.Menu(self, tearoff=0, bg=Colors.PANEL, fg=Colors.TEXT,
                       activebackground=Colors.ACCENT)
        items = self.canvas.find_closest(event.x, event.y)
        node = self._node_from_item(items[0]) if items else None
        if node:
            menu.add_command(label=f"Configure '{node.label}'",
                             command=lambda: self.on_configure_node(node) if self.on_configure_node else None)
            menu.add_command(label=f"Run '{node.label}'",
                             command=lambda: self.on_run_node(node) if self.on_run_node else None)
            menu.add_separator()
            menu.add_command(label="Delete node", command=lambda: self._delete_node(node.node_id))
        else:
            menu.add_command(label="Add Dataset", command=self._add_dataset_node)
            menu.add_command(label="Add Recipe", command=self._add_recipe_node)
            menu.add_command(label="Add Output", command=self._add_output_node)
        menu.post(event.x_root, event.y_root)

    def _select_node(self, node_id: str) -> None:
        self._deselect()
        self._selected_node = node_id
        node = self.nodes[node_id]
        if node.rect_id is not None:
            self.canvas.itemconfig(node.rect_id, outline=Colors.NODE_SELECTED, width=3)
        if self.on_node_select:
            self.on_node_select(node)

    def _deselect(self) -> None:
        if self._selected_node and self._selected_node in self.nodes:
            node = self.nodes[self._selected_node]
            if node.rect_id is not None:
                self.canvas.itemconfig(node.rect_id, outline=Colors.BORDER, width=2)
        self._selected_node = None

    def _node_from_item(self, item_id: int) -> Optional[FlowNode]:
        tags = self.canvas.gettags(item_id)
        for t in tags:
            if t in self.nodes:
                return self.nodes[t]
        return None

    # -- connect mode --------------------------------------------------------

    def _toggle_connect_mode(self) -> None:
        self._connect_mode = not self._connect_mode
        self._connect_source = None
        if self._connect_mode:
            self._connect_label.config(text="Click source node, then target node")
        else:
            self._connect_label.config(text="")

    def _handle_connect_click(self, node: FlowNode) -> None:
        if self._connect_source is None:
            self._connect_source = node.node_id
            self._connect_label.config(text=f"Source: {node.label}  → click target")
        else:
            self.add_edge(self._connect_source, node.node_id)
            self._connect_source = None
            self._connect_mode = False
            self._connect_label.config(text="")

    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add a directed edge and draw it."""
        edge = FlowEdge(source_id, target_id)
        self.edges.append(edge)
        self._redraw_edges()

    # -- delete --------------------------------------------------------------

    def _delete_selected(self) -> None:
        if self._selected_node:
            self._delete_node(self._selected_node)

    def _delete_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        for item in (node.rect_id, node.text_id, node.icon_id):
            if item is not None:
                self.canvas.delete(item)
        # Remove edges
        removed = [e for e in self.edges if e.source_id == node_id or e.target_id == node_id]
        for e in removed:
            if e.line_id is not None:
                self.canvas.delete(e.line_id)
            self.edges.remove(e)
        if self._selected_node == node_id:
            self._selected_node = None

    # -- run -----------------------------------------------------------------

    def _run_all(self) -> None:
        if self.on_run_all:
            self.on_run_all()

    # -- serialisation -------------------------------------------------------

    def get_flow_data(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def load_flow_data(self, data: Dict[str, Any]) -> None:
        """Clear canvas and rebuild from serialised data."""
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        for nd in data.get("nodes", []):
            n = FlowNode.from_dict(nd)
            self.nodes[n.node_id] = n
            self._draw_node(n)
        for ed in data.get("edges", []):
            self.edges.append(FlowEdge.from_dict(ed))
        self._redraw_edges()
