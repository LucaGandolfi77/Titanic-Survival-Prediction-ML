/* ═══════════════════════════════════════════════════════════════════════════
   operators_viz.js – Visualization operators.
   All render to a <canvas> element inside the Results panel.
   Uses Canvas 2D API – no libraries required.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

const VIZ_COLORS = [
  "#7c3aed","#06b6d4","#f59e0b","#ef4444","#10b981","#ec4899",
  "#8b5cf6","#14b8a6","#f97316","#64748b","#6366f1","#84cc16"
];

function _createChartCanvas(width = 500, height = 350) {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  c.style.background = "#1e1e2e";
  c.style.borderRadius = "8px";
  return c;
}

/* ── Data Distribution ────────────────────────────────────────────────────── */
registerOperator("Data Distribution", () => {
  const op = new Operator("Data Distribution", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("attribute", ParamKind.COLUMN, "", "Column to plot"),
      new ParamSpec("bins", ParamKind.INT, 15, "Number of bins"),
    ], "Histogram of a numeric column."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const col = this.params.attribute || df.numericCols()[0];
    const nBins = this.params.bins;
    if (!col) return { out: df };
    const vals = df.col(col).filter(v => v != null && typeof v === "number");
    const min = Math.min(...vals), max = Math.max(...vals);
    const step = (max - min) / nBins || 1;
    const bins = new Array(nBins).fill(0);
    for (const v of vals) bins[Math.min(Math.floor((v - min) / step), nBins - 1)]++;

    const c = _createChartCanvas();
    const ctx = c.getContext("2d");
    const pad = { t: 30, r: 20, b: 40, l: 50 };
    const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;
    const maxCount = Math.max(...bins, 1);
    const barW = w / nBins;

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText(`Distribution: ${col}`, pad.l, 18);

    for (let i = 0; i < nBins; i++) {
      const barH = (bins[i] / maxCount) * h;
      const x = pad.l + i * barW, y = pad.t + h - barH;
      ctx.fillStyle = VIZ_COLORS[0];
      ctx.fillRect(x + 1, y, barW - 2, barH);
    }
    // Axes
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h);
    ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();
    // Labels
    ctx.fillStyle = "#999"; ctx.font = "10px monospace";
    ctx.fillText(min.toFixed(1), pad.l, pad.t + h + 14);
    ctx.fillText(max.toFixed(1), pad.l + w - 30, pad.t + h + 14);
    ctx.fillText(String(maxCount), pad.l - 30, pad.t + 10);

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});

/* ── Scatter Plot ─────────────────────────────────────────────────────────── */
registerOperator("Scatter Plot", () => {
  const op = new Operator("Scatter Plot", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("x_attribute", ParamKind.COLUMN, "", "X axis column"),
      new ParamSpec("y_attribute", ParamKind.COLUMN, "", "Y axis column"),
      new ParamSpec("color_by", ParamKind.COLUMN, "", "Color by column (optional)"),
    ], "2D scatter plot of two attributes."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const xCol = this.params.x_attribute || df.numericCols()[0];
    const yCol = this.params.y_attribute || df.numericCols()[1] || xCol;
    const colorCol = this.params.color_by || null;

    const xVals = df.col(xCol).map(v => Number(v) || 0);
    const yVals = df.col(yCol).map(v => Number(v) || 0);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    const yMin = Math.min(...yVals), yMax = Math.max(...yVals);

    let colorVals = null, colorMap = {};
    if (colorCol) {
      colorVals = df.col(colorCol);
      const uniq = [...new Set(colorVals)].sort();
      uniq.forEach((v, i) => { colorMap[v] = VIZ_COLORS[i % VIZ_COLORS.length]; });
    }

    const c = _createChartCanvas();
    const ctx = c.getContext("2d");
    const pad = { t: 30, r: 20, b: 40, l: 50 };
    const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText(`${xCol} vs ${yCol}`, pad.l, 18);

    // Axes
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h);
    ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();

    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
    for (let i = 0; i < xVals.length; i++) {
      const px = pad.l + ((xVals[i] - xMin) / xRange) * w;
      const py = pad.t + h - ((yVals[i] - yMin) / yRange) * h;
      ctx.fillStyle = colorVals ? (colorMap[colorVals[i]] || "#7c3aed") : "#7c3aed";
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    }

    // Axis labels
    ctx.fillStyle = "#999"; ctx.font = "10px monospace";
    ctx.fillText(xMin.toFixed(1), pad.l, pad.t + h + 14);
    ctx.fillText(xMax.toFixed(1), pad.l + w - 30, pad.t + h + 14);
    ctx.fillText(yMax.toFixed(1), pad.l - 40, pad.t + 10);
    ctx.fillText(yMin.toFixed(1), pad.l - 40, pad.t + h);

    // Legend
    if (colorVals) {
      let ly = pad.t + 5;
      for (const [label, color] of Object.entries(colorMap)) {
        ctx.fillStyle = color;
        ctx.fillRect(pad.l + w - 80, ly, 10, 10);
        ctx.fillStyle = "#ccc"; ctx.font = "9px monospace";
        ctx.fillText(String(label).slice(0, 10), pad.l + w - 66, ly + 9);
        ly += 14;
      }
    }

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});

/* ── Box Plot ─────────────────────────────────────────────────────────────── */
registerOperator("Box Plot", () => {
  const op = new Operator("Box Plot", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("attributes", ParamKind.STRING, "", "Columns (comma-sep, empty=all numeric)")],
    "Box plot of numeric columns."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const cols = this.params.attributes
      ? this.params.attributes.split(",").map(s => s.trim()).filter(Boolean)
      : df.numericCols().slice(0, 6);

    const c = _createChartCanvas(Math.max(500, cols.length * 80), 350);
    const ctx = c.getContext("2d");
    const pad = { t: 30, r: 20, b: 50, l: 50 };
    const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText("Box Plot", pad.l, 18);

    const stats = df.describe();
    // Find global min/max across all displayed cols
    let gMin = Infinity, gMax = -Infinity;
    for (const c of cols) {
      if (stats[c]) { gMin = Math.min(gMin, stats[c].min); gMax = Math.max(gMax, stats[c].max); }
    }
    const range = gMax - gMin || 1;

    const boxW = Math.min(40, w / cols.length - 10);
    for (let i = 0; i < cols.length; i++) {
      const s = stats[cols[i]];
      if (!s) continue;
      const cx = pad.l + (i + 0.5) * (w / cols.length);
      const mapY = v => pad.t + h - ((v - gMin) / range) * h;

      // Whiskers
      ctx.strokeStyle = VIZ_COLORS[i % VIZ_COLORS.length]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx, mapY(s.min)); ctx.lineTo(cx, mapY(s["25%"])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, mapY(s["75%"])); ctx.lineTo(cx, mapY(s.max)); ctx.stroke();

      // Box
      const y25 = mapY(s["25%"]), y75 = mapY(s["75%"]);
      ctx.fillStyle = VIZ_COLORS[i % VIZ_COLORS.length] + "44";
      ctx.fillRect(cx - boxW / 2, y75, boxW, y25 - y75);
      ctx.strokeRect(cx - boxW / 2, y75, boxW, y25 - y75);

      // Median
      const yMed = mapY(s["50%"]);
      ctx.beginPath(); ctx.moveTo(cx - boxW / 2, yMed); ctx.lineTo(cx + boxW / 2, yMed); ctx.stroke();

      // Label
      ctx.fillStyle = "#ccc"; ctx.font = "10px monospace";
      ctx.save(); ctx.translate(cx, pad.t + h + 12);
      ctx.rotate(Math.PI / 6);
      ctx.fillText(cols[i].slice(0, 12), 0, 0);
      ctx.restore();
    }

    // Y-axis
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.stroke();

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});

/* ── Correlation Heatmap ──────────────────────────────────────────────────── */
registerOperator("Correlation Heatmap", () => {
  const op = new Operator("Correlation Heatmap", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Heatmap of pairwise correlations."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const numCols = df.numericCols().slice(0, 10);
    const n = numCols.length;
    if (n < 2) return { out: df };

    const arrs = numCols.map(c => df.col(c).map(v => v ?? 0));
    const size = 400;
    const c = _createChartCanvas(size + 80, size + 60);
    const ctx = c.getContext("2d");
    const pad = { t: 30, l: 70, r: 10, b: 30 };
    const cellW = (c.width - pad.l - pad.r) / n;
    const cellH = (c.height - pad.t - pad.b) / n;

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText("Correlation Heatmap", pad.l, 18);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const r = _pearsonR(arrs[i], arrs[j]);
        // Color: blue(-1) → white(0) → red(+1)
        const t = (r + 1) / 2;
        const red = Math.round(t * 220);
        const blue = Math.round((1 - t) * 220);
        ctx.fillStyle = `rgb(${red}, ${Math.round(80 - Math.abs(r) * 40)}, ${blue})`;
        ctx.fillRect(pad.l + j * cellW, pad.t + i * cellH, cellW - 1, cellH - 1);
        // Text
        ctx.fillStyle = "#fff"; ctx.font = `${Math.min(10, cellW / 4)}px monospace`;
        ctx.fillText(r.toFixed(2), pad.l + j * cellW + 2, pad.t + i * cellH + cellH / 2 + 3);
      }
    }

    // Labels
    ctx.fillStyle = "#ccc"; ctx.font = "9px monospace";
    for (let i = 0; i < n; i++) {
      ctx.fillText(numCols[i].slice(0, 8), 2, pad.t + i * cellH + cellH / 2 + 3);
      ctx.save(); ctx.translate(pad.l + i * cellW + cellW / 2, pad.t + n * cellH + 4);
      ctx.rotate(Math.PI / 4);
      ctx.fillText(numCols[i].slice(0, 8), 0, 0);
      ctx.restore();
    }

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});

/* ── ROC Curve ────────────────────────────────────────────────────────────── */
registerOperator("ROC Curve", () => {
  const op = new Operator("ROC Curve", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Plot ROC curve for binary classification results."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const predCol = df.columns.find(c => c.startsWith("prediction("));
    const labelColName = predCol ? predCol.match(/prediction\((.+)\)/)?.[1] : null;
    const labelCol = labelColName || df.labelCol();
    if (!predCol || !labelCol) return { out: df };

    const yTrue = df.toArray(labelCol);
    const yPred = df.toArray(predCol);
    const classes = [...new Set(yTrue)].sort();
    const posClass = classes[classes.length - 1];

    // Simple ROC: varying threshold on match score
    const trueLabels = yTrue.map(v => v === posClass ? 1 : 0);
    const predLabels = yPred.map(v => v === posClass ? 1 : 0);
    const P = trueLabels.filter(v => v === 1).length;
    const N = trueLabels.length - P;

    // Only two points for discrete predictions + origin
    let tp = 0, fp = 0;
    for (let i = 0; i < trueLabels.length; i++) {
      if (predLabels[i] === 1) { if (trueLabels[i] === 1) tp++; else fp++; }
    }
    const tpr = P ? tp / P : 0;
    const fpr = N ? fp / N : 0;
    const roc = [[0, 0], [fpr, tpr], [1, 1]];

    const c = _createChartCanvas();
    const ctx = c.getContext("2d");
    const pad = { t: 30, r: 20, b: 40, l: 50 };
    const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText(`ROC Curve (AUC ≈ ${((tpr + (1 - fpr)) / 2).toFixed(3)})`, pad.l, 18);

    // Diagonal
    ctx.strokeStyle = "#555"; ctx.setLineDash([5, 5]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t); ctx.stroke();
    ctx.setLineDash([]);

    // ROC line
    ctx.strokeStyle = VIZ_COLORS[0]; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < roc.length; i++) {
      const px = pad.l + roc[i][0] * w;
      const py = pad.t + h - roc[i][1] * h;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Axes
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h);
    ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();
    ctx.fillStyle = "#999"; ctx.font = "10px monospace";
    ctx.fillText("FPR", pad.l + w / 2, pad.t + h + 25);
    ctx.save(); ctx.translate(12, pad.t + h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("TPR", 0, 0); ctx.restore();

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});

/* ── Parallel Coordinates ─────────────────────────────────────────────────── */
registerOperator("Parallel Coordinates", () => {
  const op = new Operator("Parallel Coordinates", OpCategory.VISUALIZATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("color_by", ParamKind.COLUMN, "", "Color by column (optional)")],
    "Parallel coordinates plot of numeric features."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const numCols = df.numericCols().slice(0, 8);
    if (numCols.length < 2) return { out: df };

    const colorCol = this.params.color_by || null;
    let colorVals = null, colorMap = {};
    if (colorCol) {
      colorVals = df.col(colorCol);
      const uniq = [...new Set(colorVals)].sort();
      uniq.forEach((v, i) => { colorMap[v] = VIZ_COLORS[i % VIZ_COLORS.length]; });
    }

    const mins = numCols.map(c => { const v = df.col(c).filter(x => typeof x === "number"); return Math.min(...v); });
    const maxs = numCols.map(c => { const v = df.col(c).filter(x => typeof x === "number"); return Math.max(...v); });

    const c = _createChartCanvas(600, 350);
    const ctx = c.getContext("2d");
    const pad = { t: 30, r: 20, b: 50, l: 30 };
    const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;
    const axisSpacing = w / (numCols.length - 1);

    ctx.fillStyle = "#e0e0f0"; ctx.font = "13px monospace";
    ctx.fillText("Parallel Coordinates", pad.l, 18);

    // Axes
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    for (let a = 0; a < numCols.length; a++) {
      const x = pad.l + a * axisSpacing;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + h); ctx.stroke();
      // Label
      ctx.fillStyle = "#ccc"; ctx.font = "9px monospace";
      ctx.save(); ctx.translate(x, pad.t + h + 10); ctx.rotate(Math.PI / 6);
      ctx.fillText(numCols[a].slice(0, 10), 0, 0); ctx.restore();
    }

    // Lines (limit to 200 rows for performance)
    const nRows = Math.min(df.nRows, 200);
    ctx.globalAlpha = 0.3; ctx.lineWidth = 1;
    for (let r = 0; r < nRows; r++) {
      ctx.strokeStyle = colorVals ? (colorMap[colorVals[r]] || "#7c3aed") : "#7c3aed";
      ctx.beginPath();
      for (let a = 0; a < numCols.length; a++) {
        const ci = df.colIndex(numCols[a]);
        const v = df.data[r][ci] ?? 0;
        const range = maxs[a] - mins[a] || 1;
        const y = pad.t + h - ((v - mins[a]) / range) * h;
        const x = pad.l + a * axisSpacing;
        a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    df._vizCanvas = c;
    return { out: df };
  };
  return op;
});
