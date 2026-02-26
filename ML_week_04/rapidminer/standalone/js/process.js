/* ═══════════════════════════════════════════════════════════════════════════
   process.js – Process model, topological sort, ProcessRunner.
   Mirrors engine/process_runner.py.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Utility Operators ────────────────────────────────────────────────────── */

registerOperator("Log", () => {
  const op = new Operator("Log", OpCategory.UTILITY,
    [new Port("in", PortType.ANY)],
    [new Port("out", PortType.ANY)],
    [new ParamSpec("message", ParamKind.STRING, "", "Log message")],
    "Log a message and pass data through."
  );
  op.execute = function(inp) {
    return { out: inp.in };
  };
  return op;
});

registerOperator("Set Macro", () => {
  const op = new Operator("Set Macro", OpCategory.UTILITY,
    [],
    [],
    [
      new ParamSpec("name", ParamKind.STRING, "macro1", "Macro name"),
      new ParamSpec("value", ParamKind.STRING, "", "Macro value"),
    ], "Set a process macro variable."
  );
  op.execute = function(inp) { return {}; };
  return op;
});

registerOperator("Branch", () => {
  const op = new Operator("Branch", OpCategory.UTILITY,
    [new Port("in", PortType.ANY)],
    [new Port("then", PortType.ANY), new Port("else", PortType.ANY)],
    [
      new ParamSpec("condition_column", ParamKind.COLUMN, "", "Column to evaluate"),
      new ParamSpec("condition", ParamKind.CHOICE, "exists", "Condition", ["exists", "empty", "has_value"]),
    ], "Conditional routing."
  );
  op.execute = function(inp) {
    const df = inp.in;
    let cond = false;
    if (df instanceof DataFrame) {
      const colName = this.params.condition_column;
      switch (this.params.condition) {
        case "exists": cond = colName && df.colIndex(colName) >= 0; break;
        case "empty":  cond = df.nRows === 0; break;
        case "has_value": cond = df.nRows > 0 && colName && df.colIndex(colName) >= 0; break;
      }
    }
    return { then: cond ? df : null, else: cond ? null : df };
  };
  return op;
});

/* ── Process ──────────────────────────────────────────────────────────────── */

class Process {
  constructor(name = "Untitled") {
    this.name = name;
    this.operators = {};  // id → Operator
    this.connections = []; // Connection[]
    this.positions = {};   // id → {x, y}
    this._nextId = 1;
  }

  addOperator(opName, x = 100, y = 100) {
    const op = getOperator(opName);
    const id = `op_${this._nextId++}`;
    op.id = id;
    this.operators[id] = op;
    this.positions[id] = { x, y };
    return id;
  }

  removeOperator(id) {
    delete this.operators[id];
    delete this.positions[id];
    this.connections = this.connections.filter(c => c.fromId !== id && c.toId !== id);
  }

  connect(fromId, fromPort, toId, toPort) {
    // Validate
    if (!this.operators[fromId] || !this.operators[toId]) throw new Error("Invalid operator id");
    this.connections.push(new Connection(fromId, fromPort, toId, toPort));
  }

  disconnect(fromId, fromPort, toId, toPort) {
    this.connections = this.connections.filter(c =>
      !(c.fromId === fromId && c.fromPort === fromPort && c.toId === toId && c.toPort === toPort));
  }

  /** Kahn's algorithm topological sort */
  topologicalSort() {
    const adjOut = {};  // id → [id]
    const inDeg = {};
    for (const id of Object.keys(this.operators)) { adjOut[id] = []; inDeg[id] = 0; }
    for (const conn of this.connections) {
      if (adjOut[conn.fromId]) adjOut[conn.fromId].push(conn.toId);
      inDeg[conn.toId] = (inDeg[conn.toId] || 0) + 1;
    }
    // De-duplicate
    for (const id in adjOut) adjOut[id] = [...new Set(adjOut[id])];
    // Recount in-degree
    for (const id of Object.keys(this.operators)) inDeg[id] = 0;
    for (const conn of this.connections) inDeg[conn.toId]++;

    const queue = Object.keys(this.operators).filter(id => inDeg[id] === 0);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const next of (adjOut[id] || [])) {
        inDeg[next]--;
        if (inDeg[next] === 0) queue.push(next);
      }
    }
    if (order.length !== Object.keys(this.operators).length) {
      throw new Error("Process contains a cycle!");
    }
    return order;
  }

  /** Serialise to JSON */
  toJSON() {
    const ops = {};
    for (const [id, op] of Object.entries(this.operators)) {
      ops[id] = {
        name: op.name,
        params: { ...op.params },
        position: this.positions[id],
      };
    }
    return {
      name: this.name,
      operators: ops,
      connections: this.connections.map(c => ({
        from: c.fromId, from_port: c.fromPort,
        to: c.toId, to_port: c.toPort,
      })),
    };
  }

  /** Deserialise from JSON */
  static fromJSON(json) {
    const proc = new Process(json.name || "Untitled");
    const idMap = {};
    for (const [oldId, info] of Object.entries(json.operators)) {
      try {
        const op = getOperator(info.name);
        const id = oldId;
        op.id = id;
        Object.assign(op.params, info.params || {});
        proc.operators[id] = op;
        proc.positions[id] = info.position || { x: 100, y: 100 };
        idMap[oldId] = id;
        // Track max id
        const numPart = parseInt(id.replace("op_", ""), 10);
        if (numPart >= proc._nextId) proc._nextId = numPart + 1;
      } catch (e) {
        console.warn(`Skipping unknown operator: ${info.name}`);
      }
    }
    for (const c of (json.connections || [])) {
      const fromId = idMap[c.from] || c.from;
      const toId = idMap[c.to] || c.to;
      if (proc.operators[fromId] && proc.operators[toId]) {
        proc.connections.push(new Connection(fromId, c.from_port, toId, c.to_port));
      }
    }
    return proc;
  }
}

/* ── ProcessRunner ────────────────────────────────────────────────────────── */

class ProcessRunner {
  /**
   * @param {Process} process
   * @param {Function} onLog      – (message: string) => void
   * @param {Function} onProgress – (opId: string, opName: string, step: number, total: number) => void
   * @param {Function} onResult   – (opId: string, portName: string, data: any) => void
   */
  constructor(process, { onLog, onProgress, onResult } = {}) {
    this.process = process;
    this.onLog = onLog || (() => {});
    this.onProgress = onProgress || (() => {});
    this.onResult = onResult || (() => {});
    this._results = {};   // opId → { portName: value }
    this._stopped = false;
  }

  stop() { this._stopped = true; }

  run() {
    this._stopped = false;
    this._results = {};
    const order = this.process.topologicalSort();
    const total = order.length;

    this.onLog(`[Process] Start – ${total} operators`);
    const t0 = performance.now();

    for (let step = 0; step < total; step++) {
      if (this._stopped) { this.onLog("[Process] Stopped by user."); return this._results; }
      const opId = order[step];
      const op = this.process.operators[opId];
      this.onProgress(opId, op.name, step + 1, total);
      this.onLog(`[${step + 1}/${total}] ${op.name} (${opId})`);

      // Gather inputs
      const inputData = {};
      for (const conn of this.process.connections) {
        if (conn.toId === opId) {
          const srcOutputs = this._results[conn.fromId];
          if (srcOutputs && srcOutputs[conn.fromPort] != null) {
            inputData[conn.toPort] = srcOutputs[conn.fromPort];
          }
        }
      }

      try {
        const outputs = op.execute(inputData);
        this._results[opId] = outputs;
        // Emit results
        for (const [port, data] of Object.entries(outputs)) {
          if (data != null) this.onResult(opId, port, data);
        }
      } catch (err) {
        this.onLog(`  ✗ Error in ${op.name}: ${err.message}`);
        this._results[opId] = {};
      }
    }

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    this.onLog(`[Process] Done in ${elapsed}s`);
    return this._results;
  }

  /** Get the last result of any operator */
  getLastResult() {
    const order = Object.keys(this._results);
    if (!order.length) return null;
    const lastId = order[order.length - 1];
    return this._results[lastId];
  }
}

/* ── Sample Processes ─────────────────────────────────────────────────────── */

const SAMPLE_PROCESSES = {};

SAMPLE_PROCESSES.iris_classification = {
  name: "Iris Classification",
  operators: {
    op_1: { name: "Generate Data", params: { dataset: "iris" }, position: { x: 80, y: 120 } },
    op_2: { name: "Set Role", params: { attribute: "species", role: "label" }, position: { x: 280, y: 120 } },
    op_3: { name: "Split Data", params: { ratio: 0.7, seed: 42 }, position: { x: 480, y: 120 } },
    op_4: { name: "Random Forest", params: { n_trees: 10, max_depth: 8 }, position: { x: 680, y: 60 } },
    op_5: { name: "Apply Model", params: {}, position: { x: 880, y: 120 } },
    op_6: { name: "Performance Classification", params: {}, position: { x: 1080, y: 120 } },
  },
  connections: [
    { from: "op_1", from_port: "out", to: "op_2", to_port: "in" },
    { from: "op_2", from_port: "out", to: "op_3", to_port: "in" },
    { from: "op_3", from_port: "train", to: "op_4", to_port: "training" },
    { from: "op_4", from_port: "model", to: "op_5", to_port: "model" },
    { from: "op_3", from_port: "test", to: "op_5", to_port: "data" },
    { from: "op_5", from_port: "out", to: "op_6", to_port: "in" },
  ],
};

SAMPLE_PROCESSES.iris_clustering = {
  name: "Iris Clustering",
  operators: {
    op_1: { name: "Generate Data", params: { dataset: "iris" }, position: { x: 80, y: 120 } },
    op_2: { name: "Select Attributes", params: { attributes: "sepal_length,sepal_width,petal_length,petal_width", invert: false }, position: { x: 280, y: 120 } },
    op_3: { name: "Normalize", params: { method: "z-score" }, position: { x: 480, y: 120 } },
    op_4: { name: "KMeans", params: { k: 3 }, position: { x: 680, y: 120 } },
    op_5: { name: "Performance Clustering", params: {}, position: { x: 880, y: 120 } },
  },
  connections: [
    { from: "op_1", from_port: "out", to: "op_2", to_port: "in" },
    { from: "op_2", from_port: "out", to: "op_3", to_port: "in" },
    { from: "op_3", from_port: "out", to: "op_4", to_port: "training" },
    { from: "op_4", from_port: "out", to: "op_5", to_port: "in" },
  ],
};
