/* ═══════════════════════════════════════════════════════════════════════════
   canvas.js – Canvas rendering: operator nodes, Bézier wires, grid,
   zoom/pan, drag-drop, connection drawing via click-to-connect.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

const CANVAS_CONST = {
  NODE_W: 160,
  NODE_H: 54,
  PORT_R: 6,
  GRID_SIZE: 20,
  ZOOM_MIN: 0.3,
  ZOOM_MAX: 3,
  CATEGORY_COLORS: {
    Data: "#3b82f6", Transform: "#10b981", Feature: "#f59e0b",
    Model: "#ef4444", Evaluation: "#8b5cf6", Visualization: "#ec4899",
    Utility: "#64748b",
  },
  PORT_COLORS: {
    example_set: "#3b82f6", model: "#f59e0b",
    performance: "#10b981", any: "#94a3b8",
  },
};

class CanvasRenderer {
  constructor(canvasEl, app) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.app = app;
    // View state
    this.offsetX = 0; this.offsetY = 0; this.zoom = 1;
    // Interaction state
    this.draggingNode = null;
    this.dragOffX = 0; this.dragOffY = 0;
    this.selectedNode = null;
    this.connectingFrom = null; // { id, portName, portType, isOutput, x, y }
    this.mouseX = 0; this.mouseY = 0;
    this.isPanning = false; this.panStartX = 0; this.panStartY = 0;

    this._bindEvents();
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(canvasEl.parentElement);
    this._resize();
  }

  _resize() {
    const parent = this.canvas.parentElement;
    this.canvas.width = parent.clientWidth;
    this.canvas.height = parent.clientHeight;
    this.draw();
  }

  /* ── coordinate helpers ─────────────────────────────────────────────── */

  toWorld(ex, ey) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (ex - rect.left - this.offsetX) / this.zoom,
      y: (ey - rect.top - this.offsetY) / this.zoom,
    };
  }

  toScreen(wx, wy) {
    return {
      x: wx * this.zoom + this.offsetX,
      y: wy * this.zoom + this.offsetY,
    };
  }

  /* ── port positions ─────────────────────────────────────────────────── */

  getPortPositions(opId) {
    const op = this.app.process.operators[opId];
    const pos = this.app.process.positions[opId];
    if (!op || !pos) return { inputs: [], outputs: [] };
    const { NODE_W, NODE_H, PORT_R } = CANVAS_CONST;
    const inputs = op.inputs.map((p, i) => ({
      x: pos.x, y: pos.y + 14 + (i + 1) * (NODE_H / (op.inputs.length + 1)),
      name: p.name, type: p.type,
    }));
    const outputs = op.outputs.map((p, i) => ({
      x: pos.x + NODE_W, y: pos.y + 14 + (i + 1) * (NODE_H / (op.outputs.length + 1)),
      name: p.name, type: p.type,
    }));
    return { inputs, outputs };
  }

  /* ── hit testing ────────────────────────────────────────────────────── */

  hitTest(wx, wy) {
    const { NODE_W, NODE_H, PORT_R } = CANVAS_CONST;
    const proc = this.app.process;
    // Check ports first
    for (const [id, op] of Object.entries(proc.operators)) {
      const ports = this.getPortPositions(id);
      for (const p of ports.inputs) {
        if (Math.hypot(wx - p.x, wy - p.y) < PORT_R + 4)
          return { type: "input_port", id, port: p };
      }
      for (const p of ports.outputs) {
        if (Math.hypot(wx - p.x, wy - p.y) < PORT_R + 4)
          return { type: "output_port", id, port: p };
      }
    }
    // Check nodes (reverse order for z-ordering)
    const ids = Object.keys(proc.operators).reverse();
    for (const id of ids) {
      const pos = proc.positions[id];
      if (wx >= pos.x && wx <= pos.x + NODE_W && wy >= pos.y && wy <= pos.y + NODE_H + 14) {
        return { type: "node", id };
      }
    }
    return null;
  }

  /* ── events ─────────────────────────────────────────────────────────── */

  _bindEvents() {
    const c = this.canvas;
    c.addEventListener("mousedown", e => this._onMouseDown(e));
    c.addEventListener("mousemove", e => this._onMouseMove(e));
    c.addEventListener("mouseup",   e => this._onMouseUp(e));
    c.addEventListener("wheel",     e => this._onWheel(e), { passive: false });
    c.addEventListener("dblclick",  e => this._onDblClick(e));
    c.addEventListener("contextmenu", e => { e.preventDefault(); });

    // Drop from palette
    c.addEventListener("dragover", e => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
    c.addEventListener("drop", e => {
      e.preventDefault();
      const opName = e.dataTransfer.getData("text/plain");
      if (!opName) return;
      const { x, y } = this.toWorld(e.clientX, e.clientY);
      this.app.addOperatorToProcess(opName, x, y);
    });
  }

  _onMouseDown(e) {
    const { x, y } = this.toWorld(e.clientX, e.clientY);
    const hit = this.hitTest(x, y);

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      // Pan
      this.isPanning = true;
      this.panStartX = e.clientX - this.offsetX;
      this.panStartY = e.clientY - this.offsetY;
      return;
    }

    if (!hit) {
      this.selectedNode = null;
      this.connectingFrom = null;
      this.app.onNodeSelected(null);
      this.draw();
      return;
    }

    if (hit.type === "output_port") {
      this.connectingFrom = { id: hit.id, portName: hit.port.name, portType: hit.port.type, isOutput: true, x: hit.port.x, y: hit.port.y };
      return;
    }
    if (hit.type === "input_port") {
      this.connectingFrom = { id: hit.id, portName: hit.port.name, portType: hit.port.type, isOutput: false, x: hit.port.x, y: hit.port.y };
      return;
    }
    if (hit.type === "node") {
      this.selectedNode = hit.id;
      this.draggingNode = hit.id;
      const pos = this.app.process.positions[hit.id];
      this.dragOffX = x - pos.x;
      this.dragOffY = y - pos.y;
      this.app.onNodeSelected(hit.id);
      this.draw();
    }
  }

  _onMouseMove(e) {
    const { x, y } = this.toWorld(e.clientX, e.clientY);
    this.mouseX = x; this.mouseY = y;

    if (this.isPanning) {
      this.offsetX = e.clientX - this.panStartX;
      this.offsetY = e.clientY - this.panStartY;
      this.draw();
      return;
    }

    if (this.draggingNode) {
      const pos = this.app.process.positions[this.draggingNode];
      pos.x = x - this.dragOffX;
      pos.y = y - this.dragOffY;
      this.draw();
      return;
    }

    if (this.connectingFrom) {
      this.draw();
    }
  }

  _onMouseUp(e) {
    if (this.isPanning) { this.isPanning = false; return; }

    if (this.connectingFrom) {
      const { x, y } = this.toWorld(e.clientX, e.clientY);
      const hit = this.hitTest(x, y);
      if (hit) {
        if (this.connectingFrom.isOutput && hit.type === "input_port") {
          this.app.process.connect(this.connectingFrom.id, this.connectingFrom.portName, hit.id, hit.port.name);
          this.app.pushUndo();
        } else if (!this.connectingFrom.isOutput && hit.type === "output_port") {
          this.app.process.connect(hit.id, hit.port.name, this.connectingFrom.id, this.connectingFrom.portName);
          this.app.pushUndo();
        }
      }
      this.connectingFrom = null;
      this.draw();
    }

    this.draggingNode = null;
  }

  _onWheel(e) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.max(CANVAS_CONST.ZOOM_MIN, Math.min(CANVAS_CONST.ZOOM_MAX, this.zoom * factor));
    // Zoom towards mouse
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    this.offsetX = mx - (mx - this.offsetX) * (newZoom / this.zoom);
    this.offsetY = my - (my - this.offsetY) * (newZoom / this.zoom);
    this.zoom = newZoom;
    this.draw();
  }

  _onDblClick(e) {
    const { x, y } = this.toWorld(e.clientX, e.clientY);
    const hit = this.hitTest(x, y);
    if (hit && hit.type === "node") {
      this.app.onNodeDblClick(hit.id);
    }
  }

  /* ── drawing ────────────────────────────────────────────────────────── */

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width, h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(this.offsetX, this.offsetY);
    ctx.scale(this.zoom, this.zoom);

    this._drawGrid(w, h);
    this._drawConnections();
    this._drawNodes();
    this._drawConnectingWire();

    ctx.restore();
  }

  _drawGrid(w, h) {
    const ctx = this.ctx;
    const gs = CANVAS_CONST.GRID_SIZE;
    const startX = Math.floor(-this.offsetX / this.zoom / gs) * gs;
    const startY = Math.floor(-this.offsetY / this.zoom / gs) * gs;
    const endX = startX + w / this.zoom + gs * 2;
    const endY = startY + h / this.zoom + gs * 2;

    ctx.strokeStyle = "#2a2a3c33";
    ctx.lineWidth = 0.5;
    for (let x = startX; x < endX; x += gs) {
      ctx.beginPath(); ctx.moveTo(x, startY); ctx.lineTo(x, endY); ctx.stroke();
    }
    for (let y = startY; y < endY; y += gs) {
      ctx.beginPath(); ctx.moveTo(startX, y); ctx.lineTo(endX, y); ctx.stroke();
    }
  }

  _drawConnections() {
    const ctx = this.ctx;
    for (const conn of this.app.process.connections) {
      const fromPorts = this.getPortPositions(conn.fromId).outputs;
      const toPorts = this.getPortPositions(conn.toId).inputs;
      const fp = fromPorts.find(p => p.name === conn.fromPort);
      const tp = toPorts.find(p => p.name === conn.toPort);
      if (!fp || !tp) continue;
      this._drawBezier(fp.x, fp.y, tp.x, tp.y, CANVAS_CONST.PORT_COLORS[fp.type] || "#94a3b8");
    }
  }

  _drawBezier(x1, y1, x2, y2, color) {
    const ctx = this.ctx;
    const dx = Math.abs(x2 - x1) * 0.5;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2);
    ctx.stroke();
  }

  _drawNodes() {
    const ctx = this.ctx;
    const { NODE_W, NODE_H, PORT_R, CATEGORY_COLORS, PORT_COLORS } = CANVAS_CONST;
    const proc = this.app.process;

    for (const [id, op] of Object.entries(proc.operators)) {
      const pos = proc.positions[id];
      const catColor = CATEGORY_COLORS[op.category] || "#64748b";
      const isSelected = id === this.selectedNode;

      // Shadow
      ctx.shadowColor = "rgba(0,0,0,0.4)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 2;

      // Header
      ctx.fillStyle = catColor;
      ctx.beginPath();
      ctx.roundRect(pos.x, pos.y, NODE_W, 14, [6, 6, 0, 0]);
      ctx.fill();

      // Body
      ctx.fillStyle = isSelected ? "#3a3a5c" : "#2a2a3c";
      ctx.beginPath();
      ctx.roundRect(pos.x, pos.y + 14, NODE_W, NODE_H, [0, 0, 6, 6]);
      ctx.fill();

      ctx.shadowColor = "transparent";

      // Border
      if (isSelected) {
        ctx.strokeStyle = "#7c3aed";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(pos.x, pos.y, NODE_W, NODE_H + 14, 6);
        ctx.stroke();
      }

      // Name
      ctx.fillStyle = "#e0e0f0";
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.fillText(op.name.slice(0, 18), pos.x + 6, pos.y + 11);

      // Category tag
      ctx.fillStyle = "#999";
      ctx.font = "9px system-ui";
      ctx.fillText(op.category, pos.x + 6, pos.y + 28);

      // Ports - inputs
      for (let i = 0; i < op.inputs.length; i++) {
        const py = pos.y + 14 + (i + 1) * (NODE_H / (op.inputs.length + 1));
        ctx.fillStyle = PORT_COLORS[op.inputs[i].type] || "#94a3b8";
        ctx.beginPath(); ctx.arc(pos.x, py, PORT_R, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#ccc"; ctx.font = "8px monospace";
        ctx.fillText(op.inputs[i].name, pos.x + PORT_R + 3, py + 3);
      }

      // Ports - outputs
      for (let i = 0; i < op.outputs.length; i++) {
        const py = pos.y + 14 + (i + 1) * (NODE_H / (op.outputs.length + 1));
        ctx.fillStyle = PORT_COLORS[op.outputs[i].type] || "#94a3b8";
        ctx.beginPath(); ctx.arc(pos.x + NODE_W, py, PORT_R, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#ccc"; ctx.font = "8px monospace";
        const label = op.outputs[i].name;
        ctx.fillText(label, pos.x + NODE_W - PORT_R - ctx.measureText(label).width - 3, py + 3);
      }
    }
  }

  _drawConnectingWire() {
    if (!this.connectingFrom) return;
    const cf = this.connectingFrom;
    if (cf.isOutput) {
      this._drawBezier(cf.x, cf.y, this.mouseX, this.mouseY, "#7c3aed88");
    } else {
      this._drawBezier(this.mouseX, this.mouseY, cf.x, cf.y, "#7c3aed88");
    }
  }

  /* ── utilities ──────────────────────────────────────────────────────── */

  fitView() {
    const proc = this.app.process;
    const ids = Object.keys(proc.operators);
    if (!ids.length) { this.offsetX = 0; this.offsetY = 0; this.zoom = 1; this.draw(); return; }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const id of ids) {
      const p = proc.positions[id];
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + CANVAS_CONST.NODE_W);
      maxY = Math.max(maxY, p.y + CANVAS_CONST.NODE_H + 14);
    }
    const pw = maxX - minX + 100, ph = maxY - minY + 100;
    const z = Math.min(this.canvas.width / pw, this.canvas.height / ph, 1.5);
    this.zoom = z;
    this.offsetX = (this.canvas.width - pw * z) / 2 - minX * z + 50 * z;
    this.offsetY = (this.canvas.height - ph * z) / 2 - minY * z + 50 * z;
    this.draw();
  }
}
