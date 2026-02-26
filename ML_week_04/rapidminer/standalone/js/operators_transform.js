/* ═══════════════════════════════════════════════════════════════════════════
   operators_transform.js – 17 data-transformation operators.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Select Attributes ────────────────────────────────────────────────────── */
registerOperator("Select Attributes", () => {
  const op = new Operator("Select Attributes", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attributes", ParamKind.STRING, "", "Comma-separated column names to keep"),
      new ParamSpec("invert", ParamKind.BOOL, false, "Invert selection (drop listed columns)"),
    ], "Select or remove columns.");
  op.execute = function(inp) {
    const df = inp.in;
    const attrs = this.params.attributes.split(",").map(s => s.trim()).filter(Boolean);
    if (!attrs.length) return { out: df };
    return { out: this.params.invert ? df.drop(attrs) : df.select(attrs) };
  };
  return op;
});

/* ── Filter Examples ──────────────────────────────────────────────────────── */
registerOperator("Filter Examples", () => {
  const op = new Operator("Filter Examples", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("condition", ParamKind.STRING, "", "Filter expression, e.g. age > 30")],
    "Filter rows based on a condition.");
  op.execute = function(inp) {
    return { out: inp.in.filterExpr(this.params.condition) };
  };
  return op;
});

/* ── Rename ───────────────────────────────────────────────────────────────── */
registerOperator("Rename", () => {
  const op = new Operator("Rename", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("old_name", ParamKind.COLUMN, "", "Current column name"),
      new ParamSpec("new_name", ParamKind.STRING, "", "New column name"),
    ], "Rename a single column.");
  op.execute = function(inp) {
    return { out: inp.in.rename({ [this.params.old_name]: this.params.new_name }) };
  };
  return op;
});

/* ── Set Role ─────────────────────────────────────────────────────────────── */
registerOperator("Set Role", () => {
  const op = new Operator("Set Role", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attribute", ParamKind.COLUMN, "", "Column"),
      new ParamSpec("role", ParamKind.CHOICE, "label", "Role", ["label", "id", "weight", "prediction", "regular"]),
    ], "Assign a special role to a column.");
  op.execute = function(inp) {
    return { out: inp.in.setRole(this.params.attribute, this.params.role) };
  };
  return op;
});

/* ── Replace Missing ─────────────────────────────────────────────────────── */
registerOperator("Replace Missing", () => {
  const op = new Operator("Replace Missing", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("strategy", ParamKind.CHOICE, "mean", "Strategy", ["mean","median","mode","constant"]),
      new ParamSpec("constant", ParamKind.FLOAT, 0, "Constant value (if strategy=constant)"),
    ], "Replace null / NaN values.");
  op.execute = function(inp) {
    return { out: inp.in.fillNA(this.params.strategy, this.params.constant) };
  };
  return op;
});

/* ── Normalize ────────────────────────────────────────────────────────────── */
registerOperator("Normalize", () => {
  const op = new Operator("Normalize", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("method", ParamKind.CHOICE, "z-score", "Method", ["z-score","min-max","log"]),
    ], "Normalize numeric columns.");
  op.execute = function(inp) {
    return { out: inp.in.normalize(this.params.method) };
  };
  return op;
});

/* ── Discretize ───────────────────────────────────────────────────────────── */
registerOperator("Discretize", () => {
  const op = new Operator("Discretize", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attribute", ParamKind.COLUMN, "", "Column to discretize"),
      new ParamSpec("bins", ParamKind.INT, 5, "Number of bins"),
    ], "Discretize a numeric column into bins.");
  op.execute = function(inp) {
    const df = inp.in.clone();
    const col = this.params.attribute;
    const nBins = this.params.bins;
    const ci = df.colIndex(col);
    if (ci < 0) return { out: df };
    const vals = df.data.map(r => r[ci]).filter(v => v != null && typeof v === "number");
    if (!vals.length) return { out: df };
    const min = Math.min(...vals), max = Math.max(...vals);
    const step = (max - min) / nBins || 1;
    for (const row of df.data) {
      if (row[ci] != null && typeof row[ci] === "number") {
        const bin = Math.min(Math.floor((row[ci] - min) / step), nBins - 1);
        const lo = (min + bin * step).toFixed(2);
        const hi = (min + (bin + 1) * step).toFixed(2);
        row[ci] = `[${lo}, ${hi})`;
      }
    }
    return { out: df };
  };
  return op;
});

/* ── Generate Attributes ──────────────────────────────────────────────────── */
registerOperator("Generate Attributes", () => {
  const op = new Operator("Generate Attributes", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("name", ParamKind.STRING, "new_attr", "New column name"),
      new ParamSpec("expression", ParamKind.STRING, "", "Expression, e.g. col1 + col2"),
    ], "Create a new attribute from an expression.");
  op.execute = function(inp) {
    const df = inp.in;
    const expr = this.params.expression;
    const colName = this.params.name;
    // Build a simple evaluator using column names as variables
    const values = df.data.map((row, ri) => {
      try {
        let e = expr;
        for (let ci = 0; ci < df.columns.length; ci++) {
          const cName = df.columns[ci];
          const re = new RegExp(`\\b${cName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, "g");
          const val = row[ci] ?? 0;
          e = e.replace(re, typeof val === "number" ? val : `"${val}"`);
        }
        return Function(`"use strict"; return (${e});`)();
      } catch { return null; }
    });
    return { out: df.addColumn(colName, values) };
  };
  return op;
});

/* ── Remove Duplicates ────────────────────────────────────────────────────── */
registerOperator("Remove Duplicates", () => {
  const op = new Operator("Remove Duplicates", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("attributes", ParamKind.STRING, "", "Subset columns (comma-sep, empty=all)")],
    "Remove duplicate rows.");
  op.execute = function(inp) {
    const subset = this.params.attributes ? this.params.attributes.split(",").map(s=>s.trim()).filter(Boolean) : null;
    return { out: inp.in.removeDuplicates(subset.length ? subset : null) };
  };
  return op;
});

/* ── Sort ─────────────────────────────────────────────────────────────────── */
registerOperator("Sort", () => {
  const op = new Operator("Sort", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attribute", ParamKind.COLUMN, "", "Sort column"),
      new ParamSpec("ascending", ParamKind.BOOL, true, "Ascending order"),
    ], "Sort rows by a column.");
  op.execute = function(inp) {
    return { out: inp.in.sort(this.params.attribute, this.params.ascending) };
  };
  return op;
});

/* ── Sample ───────────────────────────────────────────────────────────────── */
registerOperator("Sample", () => {
  const op = new Operator("Sample", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("size", ParamKind.INT, 100, "Number of rows"),
      new ParamSpec("seed", ParamKind.INT, 42, "Random seed"),
    ], "Random sample of rows.");
  op.execute = function(inp) {
    return { out: inp.in.sample(this.params.size, this.params.seed) };
  };
  return op;
});

/* ── Split Data ───────────────────────────────────────────────────────────── */
registerOperator("Split Data", () => {
  const op = new Operator("Split Data", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("train", PortType.EXAMPLE_SET), new Port("test", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("ratio", ParamKind.FLOAT, 0.7, "Train fraction"),
      new ParamSpec("seed", ParamKind.INT, 42, "Random seed"),
    ], "Split data into train and test sets.");
  op.execute = function(inp) {
    const { train, test } = inp.in.trainTestSplit(this.params.ratio, this.params.seed);
    return { train, test };
  };
  return op;
});

/* ── Append ───────────────────────────────────────────────────────────────── */
registerOperator("Append", () => {
  const op = new Operator("Append", OpCategory.TRANSFORM,
    [new Port("in1", PortType.EXAMPLE_SET), new Port("in2", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Concatenate two datasets vertically.");
  op.execute = function(inp) {
    return { out: inp.in1.concat(inp.in2) };
  };
  return op;
});

/* ── Join ─────────────────────────────────────────────────────────────────── */
registerOperator("Join", () => {
  const op = new Operator("Join", OpCategory.TRANSFORM,
    [new Port("left", PortType.EXAMPLE_SET), new Port("right", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("key", ParamKind.STRING, "", "Join key column(s), comma-separated"),
      new ParamSpec("type", ParamKind.CHOICE, "inner", "Join type", ["inner","left","right","outer"]),
    ], "Join two datasets on key columns.");
  op.execute = function(inp) {
    const keys = this.params.key.split(",").map(s => s.trim()).filter(Boolean);
    return { out: inp.left.join(inp.right, keys, this.params.type) };
  };
  return op;
});

/* ── Aggregate ────────────────────────────────────────────────────────────── */
registerOperator("Aggregate", () => {
  const op = new Operator("Aggregate", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("group_by", ParamKind.COLUMN, "", "Group-by column"),
      new ParamSpec("aggregation", ParamKind.CHOICE, "mean", "Aggregation function", ["mean","sum","count","min","max"]),
    ], "Aggregate data by group.");
  op.execute = function(inp) {
    return { out: inp.in.aggregate(this.params.group_by, this.params.aggregation) };
  };
  return op;
});

/* ── Transpose ────────────────────────────────────────────────────────────── */
registerOperator("Transpose", () => {
  const op = new Operator("Transpose", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Transpose the dataset (swap rows ↔ columns).");
  op.execute = function(inp) {
    return { out: inp.in.transpose() };
  };
  return op;
});

/* ── Pivot ────────────────────────────────────────────────────────────────── */
registerOperator("Pivot", () => {
  const op = new Operator("Pivot", OpCategory.TRANSFORM,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("index", ParamKind.COLUMN, "", "Row index column"),
      new ParamSpec("columns", ParamKind.COLUMN, "", "Pivot columns column"),
      new ParamSpec("values", ParamKind.COLUMN, "", "Values column"),
      new ParamSpec("aggfunc", ParamKind.CHOICE, "sum", "Aggregation", ["sum","mean","count","min","max"]),
    ], "Pivot table.");
  op.execute = function(inp) {
    return { out: inp.in.pivot(this.params.index, this.params.columns, this.params.values, this.params.aggfunc) };
  };
  return op;
});
