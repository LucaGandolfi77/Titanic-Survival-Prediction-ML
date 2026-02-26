/* ═══════════════════════════════════════════════════════════════════════════
   operators_feature.js – Feature engineering operators.
   PCA, Variance Threshold, Correlation Matrix, Forward Selection,
   Backward Elimination, Weight by Correlation, One Hot Encoding,
   Label Encoding, Target Encoding.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── PCA Operator ─────────────────────────────────────────────────────────── */
registerOperator("PCA", () => {
  const op = new Operator("PCA", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("n_components", ParamKind.INT, 2, "Number of principal components")],
    "Dimensionality reduction via Principal Component Analysis."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const numCols = df.numericCols();
    const X = df.toMatrix(numCols);
    const pca = new PCA({ nComponents: this.params.n_components });
    const projected = pca.fitTransform(X);
    const newCols = [];
    for (let i = 0; i < this.params.n_components; i++) newCols.push(`PC${i + 1}`);
    // Keep non-numeric columns + add PCs
    const catCols = df.columns.filter(c => !numCols.includes(c));
    const catIdxs = catCols.map(c => df.colIndex(c));
    const allCols = [...catCols, ...newCols];
    const data = df.data.map((row, ri) => [
      ...catIdxs.map(ci => row[ci]),
      ...projected[ri],
    ]);
    return { out: new DataFrame(allCols, data, { ...df.roles }) };
  };
  return op;
});

/* ── Variance Threshold ───────────────────────────────────────────────────── */
registerOperator("Variance Threshold", () => {
  const op = new Operator("Variance Threshold", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("threshold", ParamKind.FLOAT, 0.0, "Minimum variance to keep a feature")],
    "Remove features with variance below threshold."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const thr = this.params.threshold;
    const keep = df.columns.filter(c => {
      const vals = df.col(c).filter(v => v != null && typeof v === "number");
      if (!vals.length) return true; // keep non-numeric
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const vari = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
      return vari >= thr;
    });
    return { out: df.select(keep) };
  };
  return op;
});

/* ── Forward Selection ────────────────────────────────────────────────────── */
registerOperator("Forward Selection", () => {
  const op = new Operator("Forward Selection", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("max_features", ParamKind.INT, 5, "Max features to select")],
    "Select top features by correlation with label."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const labelCol = df.labelCol();
    const labelIdx = df.colIndex(labelCol);
    const labelVals = df.data.map(r => r[labelIdx]);
    const numLabel = labelVals.every(v => typeof v === "number");
    if (!numLabel) return { out: df }; // can't evaluate correlation with categorical label here
    const numCols = df.numericCols().filter(c => c !== labelCol);
    // Rank by absolute correlation with label
    const scores = numCols.map(c => {
      const vals = df.col(c).map(v => v ?? 0);
      return { col: c, score: Math.abs(_pearsonR(vals, labelVals)) };
    });
    scores.sort((a, b) => b.score - a.score);
    const selected = scores.slice(0, this.params.max_features).map(s => s.col);
    // Keep label + selected
    const keep = [...selected, labelCol];
    const catCols = df.columns.filter(c => !df.numericCols().includes(c) && c !== labelCol);
    return { out: df.select([...catCols, ...keep]) };
  };
  return op;
});

/* ── Backward Elimination ─────────────────────────────────────────────────── */
registerOperator("Backward Elimination", () => {
  const op = new Operator("Backward Elimination", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("min_correlation", ParamKind.FLOAT, 0.1, "Remove features with |corr| < threshold")],
    "Remove features weakly correlated with label."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const labelCol = df.labelCol();
    const labelVals = df.col(labelCol).map(v => (typeof v === "number" ? v : 0));
    const keep = df.columns.filter(c => {
      if (c === labelCol) return true;
      const vals = df.col(c);
      if (!vals.every(v => v == null || typeof v === "number")) return true;
      const numVals = vals.map(v => v ?? 0);
      return Math.abs(_pearsonR(numVals, labelVals)) >= this.params.min_correlation;
    });
    return { out: df.select(keep) };
  };
  return op;
});

/* ── Correlation Matrix ───────────────────────────────────────────────────── */
registerOperator("Correlation Matrix", () => {
  const op = new Operator("Correlation Matrix", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Compute the correlation matrix of numeric features."
  );
  op.execute = function(inp) {
    return { out: inp.in.corr() };
  };
  return op;
});

/* ── Weight by Correlation ────────────────────────────────────────────────── */
registerOperator("Weight by Correlation", () => {
  const op = new Operator("Weight by Correlation", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Compute feature weights based on correlation with label."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const labelCol = df.labelCol();
    const labelVals = df.col(labelCol).map(v => (typeof v === "number" ? v : 0));
    const numCols = df.numericCols().filter(c => c !== labelCol);
    const weights = numCols.map(c => {
      const vals = df.col(c).map(v => v ?? 0);
      return [c, Math.abs(_pearsonR(vals, labelVals))];
    });
    weights.sort((a, b) => b[1] - a[1]);
    return { out: new DataFrame(["feature", "weight"], weights) };
  };
  return op;
});

/* ── One Hot Encoding ─────────────────────────────────────────────────────── */
registerOperator("One Hot Encoding", () => {
  const op = new Operator("One Hot Encoding", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attributes", ParamKind.STRING, "", "Columns to encode (comma-sep, empty=all categorical)"),
      new ParamSpec("drop_first", ParamKind.BOOL, false, "Drop first category to avoid multicollinearity"),
    ], "One-hot encode categorical columns."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const colList = this.params.attributes
      ? this.params.attributes.split(",").map(s => s.trim()).filter(Boolean)
      : df.categoricalCols();
    return { out: df.oneHotEncode(colList, this.params.drop_first) };
  };
  return op;
});

/* ── Label Encoding ───────────────────────────────────────────────────────── */
registerOperator("Label Encoding", () => {
  const op = new Operator("Label Encoding", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("attributes", ParamKind.STRING, "", "Columns to encode (comma-sep, empty=all categorical)")],
    "Map categorical values to integers."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const colList = this.params.attributes
      ? this.params.attributes.split(",").map(s => s.trim()).filter(Boolean)
      : df.categoricalCols();
    const { df: encoded } = df.labelEncode(colList);
    return { out: encoded };
  };
  return op;
});

/* ── Target Encoding ──────────────────────────────────────────────────────── */
registerOperator("Target Encoding", () => {
  const op = new Operator("Target Encoding", OpCategory.FEATURE,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("attributes", ParamKind.STRING, "", "Columns to encode (comma-sep)")],
    "Replace categorical values with mean of target (label) per category."
  );
  op.execute = function(inp) {
    const df = inp.in.clone();
    const labelCol = df.labelCol();
    const labelIdx = df.colIndex(labelCol);
    const colList = this.params.attributes
      ? this.params.attributes.split(",").map(s => s.trim()).filter(Boolean)
      : df.categoricalCols().filter(c => c !== labelCol);
    for (const c of colList) {
      const ci = df.colIndex(c);
      if (ci < 0) continue;
      // Compute mean target per category
      const groups = {};
      for (const row of df.data) {
        const cat = row[ci];
        if (cat == null) continue;
        if (!groups[cat]) groups[cat] = [];
        const lv = row[labelIdx];
        if (typeof lv === "number") groups[cat].push(lv);
      }
      const means = {};
      for (const [k, vals] of Object.entries(groups)) {
        means[k] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      }
      for (const row of df.data) {
        if (row[ci] != null && means[row[ci]] != null) row[ci] = means[row[ci]];
      }
    }
    return { out: df };
  };
  return op;
});
