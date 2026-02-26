/* ═══════════════════════════════════════════════════════════════════════════
   operators.js – Base classes, typed ports, registry, and Connection.
   Mirrors engine/operator_base.py from the Python project.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Enums ────────────────────────────────────────────────────────────────── */

const PortType = Object.freeze({
  EXAMPLE_SET: "example_set",
  MODEL:       "model",
  PERFORMANCE: "performance",
  ANY:         "any",
});

const OpCategory = Object.freeze({
  DATA:          "Data",
  TRANSFORM:     "Transform",
  FEATURE:       "Feature",
  MODEL:         "Model",
  EVALUATION:    "Evaluation",
  VISUALIZATION: "Visualization",
  UTILITY:       "Utility",
});

const ParamKind = Object.freeze({
  INT:    "int",
  FLOAT:  "float",
  STRING: "string",
  BOOL:   "bool",
  CHOICE: "choice",
  COLUMN: "column",
});

/* ── ParamSpec ────────────────────────────────────────────────────────────── */

class ParamSpec {
  /**
   * @param {string} name
   * @param {string} kind       – one of ParamKind values
   * @param {*}      defaultVal
   * @param {string} description
   * @param {string[]} choices  – only for ParamKind.CHOICE
   */
  constructor(name, kind, defaultVal, description = "", choices = []) {
    this.name = name;
    this.kind = kind;
    this.default = defaultVal;
    this.description = description;
    this.choices = choices;
  }
}

/* ── Port ─────────────────────────────────────────────────────────────────── */

class Port {
  constructor(name, type = PortType.ANY, multi = false) {
    this.name = name;
    this.type = type;
    this.multi = multi;
  }
}

/* ── Operator (abstract base) ─────────────────────────────────────────────── */

class Operator {
  /**
   * @param {string}   name
   * @param {string}   category – one of OpCategory values
   * @param {Port[]}   inputs
   * @param {Port[]}   outputs
   * @param {ParamSpec[]} params
   * @param {string}   description
   */
  constructor(name, category, inputs, outputs, params = [], description = "") {
    this.name = name;
    this.category = category;
    this.inputs = inputs;
    this.outputs = outputs;
    this.paramSpecs = params;
    this.description = description;
    // Current parameter values
    this.params = {};
    for (const p of params) this.params[p.name] = p.default;
    // Runtime id assigned when placed on canvas
    this.id = null;
  }

  /**
   * Execute the operator.
   * @param {Object} inputData – { portName: value }
   * @returns {Object}         – { portName: value }
   */
  execute(inputData) {
    throw new Error(`Operator '${this.name}' must implement execute()`);
  }

  /** Clone parameters into a fresh copy of this operator */
  clone() {
    const op = Object.create(Object.getPrototypeOf(this));
    Object.assign(op, this);
    op.params = { ...this.params };
    op.id = null;
    return op;
  }
}

/* ── Connection ───────────────────────────────────────────────────────────── */

class Connection {
  constructor(fromId, fromPort, toId, toPort) {
    this.fromId = fromId;
    this.fromPort = fromPort;
    this.toId = toId;
    this.toPort = toPort;
  }
}

/* ── Operator Registry ────────────────────────────────────────────────────── */

const _operatorRegistry = {};

/**
 * Register an operator class.
 * @param {string} name
 * @param {Function} factory – () => Operator
 */
function registerOperator(name, factory) {
  _operatorRegistry[name] = factory;
}

/** Get a fresh instance of a registered operator */
function getOperator(name) {
  const factory = _operatorRegistry[name];
  if (!factory) throw new Error(`Unknown operator: '${name}'`);
  return factory();
}

/** List all registered operator names */
function listOperators() {
  return Object.keys(_operatorRegistry).sort();
}

/** Map category → [operatorName, ...] */
function operatorsByCategory() {
  const map = {};
  for (const name of listOperators()) {
    const op = getOperator(name);
    const cat = op.category;
    if (!map[cat]) map[cat] = [];
    map[cat].push(name);
  }
  return map;
}
