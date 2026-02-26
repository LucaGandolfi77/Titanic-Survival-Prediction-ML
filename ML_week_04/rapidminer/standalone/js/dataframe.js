/* ═══════════════════════════════════════════════════════════════════════════
   dataframe.js – Lightweight tabular data structure for RapidMiner Lite.
   Mirrors pandas DataFrame: column-oriented storage, CSV parse, filter,
   sort, aggregate, join, transpose, pivot, stats.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

class DataFrame {
  /**
   * @param {string[]} columns
   * @param {any[][]}  data  – array of row arrays
   * @param {Object}   roles – e.g. { species: "label" }
   */
  constructor(columns = [], data = [], roles = {}) {
    this.columns = [...columns];
    this.data = data.map(r => [...r]);
    this.roles = { ...roles };
  }

  /* ── factories ──────────────────────────────────────────────────────── */

  static fromCSV(text, sep = ",", header = true) {
    const lines = text.replace(/\r\n/g, "\n").split("\n").filter(l => l.trim());
    if (!lines.length) return new DataFrame();
    const split = l => {
      const out = []; let cur = "", inQ = false;
      for (let i = 0; i < l.length; i++) {
        const ch = l[i];
        if (ch === '"') { inQ = !inQ; continue; }
        if (ch === sep && !inQ) { out.push(cur); cur = ""; continue; }
        cur += ch;
      }
      out.push(cur);
      return out;
    };
    let cols, startRow;
    if (header) {
      cols = split(lines[0]).map(c => c.trim());
      startRow = 1;
    } else {
      const first = split(lines[0]);
      cols = first.map((_, i) => `col_${i}`);
      startRow = 0;
    }
    const rows = [];
    for (let i = startRow; i < lines.length; i++) {
      const vals = split(lines[i]);
      const row = vals.map(v => {
        v = v.trim();
        if (v === "" || v === "NA" || v === "NaN" || v === "null") return null;
        const n = Number(v);
        return isNaN(n) ? v : n;
      });
      while (row.length < cols.length) row.push(null);
      rows.push(row.slice(0, cols.length));
    }
    return new DataFrame(cols, rows);
  }

  static fromJSON(arr) {
    if (!arr.length) return new DataFrame();
    const cols = Object.keys(arr[0]);
    const data = arr.map(o => cols.map(c => o[c] ?? null));
    return new DataFrame(cols, data);
  }

  static fromObjects(arr) { return DataFrame.fromJSON(arr); }

  /* ── serialisation ──────────────────────────────────────────────────── */

  toCSV(sep = ",") {
    const esc = v => {
      if (v == null) return "";
      const s = String(v);
      return s.includes(sep) || s.includes('"') || s.includes("\n")
        ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const hdr = this.columns.map(esc).join(sep);
    const body = this.data.map(r => r.map(esc).join(sep)).join("\n");
    return hdr + "\n" + body;
  }

  toJSON() {
    return this.data.map(row => {
      const obj = {};
      this.columns.forEach((c, i) => { obj[c] = row[i]; });
      return obj;
    });
  }

  /* ── accessors ──────────────────────────────────────────────────────── */

  get nRows() { return this.data.length; }
  get nCols() { return this.columns.length; }
  get shape() { return [this.nRows, this.nCols]; }

  colIndex(name) { return this.columns.indexOf(name); }

  col(name) {
    const idx = this.colIndex(name);
    if (idx < 0) throw new Error(`Column '${name}' not found`);
    return this.data.map(r => r[idx]);
  }

  numericCols() {
    return this.columns.filter(c => {
      const vals = this.col(c).filter(v => v != null);
      return vals.length > 0 && vals.every(v => typeof v === "number");
    });
  }

  categoricalCols() {
    return this.columns.filter(c => !this.numericCols().includes(c));
  }

  row(i) {
    const obj = {};
    this.columns.forEach((c, j) => { obj[c] = this.data[i][j]; });
    return obj;
  }

  head(n = 5) {
    return new DataFrame(this.columns, this.data.slice(0, n), { ...this.roles });
  }

  tail(n = 5) {
    return new DataFrame(this.columns, this.data.slice(-n), { ...this.roles });
  }

  clone() {
    return new DataFrame(this.columns, this.data, { ...this.roles });
  }

  /* ── column operations ──────────────────────────────────────────────── */

  select(colNames) {
    const idxs = colNames.map(c => this.colIndex(c)).filter(i => i >= 0);
    const cols = idxs.map(i => this.columns[i]);
    const data = this.data.map(r => idxs.map(i => r[i]));
    const roles = {};
    for (const [k, v] of Object.entries(this.roles)) if (cols.includes(k)) roles[k] = v;
    return new DataFrame(cols, data, roles);
  }

  drop(colNames) {
    const keep = this.columns.filter(c => !colNames.includes(c));
    return this.select(keep);
  }

  rename(mapping) {
    const df = this.clone();
    for (const [old, nw] of Object.entries(mapping)) {
      const i = df.colIndex(old);
      if (i >= 0) {
        df.columns[i] = nw;
        if (df.roles[old]) { df.roles[nw] = df.roles[old]; delete df.roles[old]; }
      }
    }
    return df;
  }

  addColumn(name, values) {
    const df = this.clone();
    df.columns.push(name);
    df.data.forEach((r, i) => r.push(values[i] ?? null));
    return df;
  }

  setRole(column, role) {
    const df = this.clone();
    df.roles[column] = role;
    return df;
  }

  labelCol() {
    for (const [c, r] of Object.entries(this.roles)) if (r === "label") return c;
    return this.columns[this.columns.length - 1];
  }

  /* ── filtering / sorting ────────────────────────────────────────────── */

  filter(fn) {
    const data = this.data.filter((row, i) => {
      const obj = {};
      this.columns.forEach((c, j) => { obj[c] = row[j]; });
      return fn(obj, i);
    });
    return new DataFrame(this.columns, data, { ...this.roles });
  }

  filterExpr(expr) {
    if (!expr || !expr.trim()) return this.clone();
    // Simple expression evaluator: column op value
    // Supports: ==, !=, >, <, >=, <=, includes
    const m = expr.match(/^\s*(\w+)\s*(==|!=|>=|<=|>|<|includes)\s*(.+)\s*$/);
    if (!m) return this.clone();
    const [, col, op, rawVal] = m;
    const idx = this.colIndex(col);
    if (idx < 0) return this.clone();
    let val = rawVal.replace(/^["']|["']$/g, "").trim();
    const numVal = Number(val);
    const isNum = !isNaN(numVal);
    const ops = {
      "==": (a, b) => a == b,
      "!=": (a, b) => a != b,
      ">":  (a, b) => a > b,
      "<":  (a, b) => a < b,
      ">=": (a, b) => a >= b,
      "<=": (a, b) => a <= b,
      "includes": (a, b) => String(a).includes(b),
    };
    const data = this.data.filter(r => {
      const v = r[idx];
      if (v == null) return false;
      const cmp = isNum && typeof v === "number" ? numVal : val;
      return ops[op](v, cmp);
    });
    return new DataFrame(this.columns, data, { ...this.roles });
  }

  sort(col, ascending = true) {
    const idx = this.colIndex(col);
    if (idx < 0) return this.clone();
    const df = this.clone();
    df.data.sort((a, b) => {
      let va = a[idx], vb = b[idx];
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") return ascending ? va - vb : vb - va;
      return ascending ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
    return df;
  }

  removeDuplicates(subsetCols = null) {
    const cols = subsetCols || this.columns;
    const idxs = cols.map(c => this.colIndex(c)).filter(i => i >= 0);
    const seen = new Set();
    const data = this.data.filter(r => {
      const key = idxs.map(i => JSON.stringify(r[i])).join("|");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    return new DataFrame(this.columns, data, { ...this.roles });
  }

  sample(n, seed = 42) {
    const rng = _seedRng(seed);
    const shuffled = [...this.data].sort(() => rng() - 0.5);
    return new DataFrame(this.columns, shuffled.slice(0, Math.min(n, shuffled.length)), { ...this.roles });
  }

  sampleFraction(frac, seed = 42) {
    return this.sample(Math.round(this.nRows * frac), seed);
  }

  /* ── split ──────────────────────────────────────────────────────────── */

  trainTestSplit(ratio = 0.7, seed = 42) {
    const rng = _seedRng(seed);
    const idxs = this.data.map((_, i) => i).sort(() => rng() - 0.5);
    const cut = Math.round(idxs.length * ratio);
    const trainIdx = idxs.slice(0, cut);
    const testIdx = idxs.slice(cut);
    return {
      train: new DataFrame(this.columns, trainIdx.map(i => this.data[i]), { ...this.roles }),
      test:  new DataFrame(this.columns, testIdx.map(i => this.data[i]),  { ...this.roles }),
    };
  }

  /* ── aggregation ────────────────────────────────────────────────────── */

  groupBy(groupCols, aggDict) {
    // aggDict: { colName: "mean" | "sum" | "count" | "min" | "max" }
    const gIdxs = groupCols.map(c => this.colIndex(c));
    const groups = new Map();
    for (const row of this.data) {
      const key = gIdxs.map(i => row[i]).join("|||");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    const aggCols = Object.keys(aggDict);
    const newCols = [...groupCols, ...aggCols.map(c => `${aggDict[c]}_${c}`)];
    const newData = [];
    for (const [, rows] of groups) {
      const gVals = gIdxs.map(i => rows[0][i]);
      const aVals = aggCols.map(c => {
        const ci = this.colIndex(c);
        const vals = rows.map(r => r[ci]).filter(v => v != null && typeof v === "number");
        switch (aggDict[c]) {
          case "mean": return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
          case "sum":  return vals.reduce((a, b) => a + b, 0);
          case "count": return rows.length;
          case "min":  return vals.length ? Math.min(...vals) : null;
          case "max":  return vals.length ? Math.max(...vals) : null;
          default: return null;
        }
      });
      newData.push([...gVals, ...aVals]);
    }
    return new DataFrame(newCols, newData);
  }

  aggregate(groupCol, agg = "mean") {
    const gi = this.colIndex(groupCol);
    if (gi < 0) return this.clone();
    const numCols = this.numericCols();
    const aggDict = {};
    for (const c of numCols) aggDict[c] = agg;
    return this.groupBy([groupCol], aggDict);
  }

  /* ── join ────────────────────────────────────────────────────────────── */

  join(other, keyCols, type = "inner") {
    const lIdxs = keyCols.map(c => this.colIndex(c));
    const rIdxs = keyCols.map(c => other.colIndex(c));
    const rExtraCols = other.columns.filter(c => !keyCols.includes(c));
    const rExtraIdxs = rExtraCols.map(c => other.colIndex(c));
    const newCols = [...this.columns, ...rExtraCols];

    const rMap = new Map();
    for (const row of other.data) {
      const key = rIdxs.map(i => row[i]).join("|||");
      if (!rMap.has(key)) rMap.set(key, []);
      rMap.get(key).push(row);
    }

    const data = [];
    const usedKeys = new Set();
    for (const lRow of this.data) {
      const key = lIdxs.map(i => lRow[i]).join("|||");
      const rRows = rMap.get(key);
      if (rRows) {
        usedKeys.add(key);
        for (const rRow of rRows) {
          data.push([...lRow, ...rExtraIdxs.map(i => rRow[i])]);
        }
      } else if (type === "left" || type === "outer") {
        data.push([...lRow, ...rExtraCols.map(() => null)]);
      }
    }
    if (type === "right" || type === "outer") {
      for (const rRow of other.data) {
        const key = rIdxs.map(i => rRow[i]).join("|||");
        if (!usedKeys.has(key)) {
          const lNulls = this.columns.map((c, i) =>
            keyCols.includes(c) ? rRow[rIdxs[keyCols.indexOf(c)]] : null);
          data.push([...lNulls, ...rExtraIdxs.map(i => rRow[i])]);
        }
      }
    }
    return new DataFrame(newCols, data);
  }

  /* ── reshape ────────────────────────────────────────────────────────── */

  transpose() {
    const numCols = this.numericCols();
    const idxs = numCols.map(c => this.colIndex(c));
    const newCols = ["feature", ...this.data.map((_, i) => `row_${i}`)];
    const newData = idxs.map(ci => [this.columns[ci], ...this.data.map(r => r[ci])]);
    return new DataFrame(newCols, newData);
  }

  pivot(indexCol, columnsCol, valuesCol, aggfunc = "sum") {
    const iIdx = this.colIndex(indexCol);
    const cIdx = this.colIndex(columnsCol);
    const vIdx = this.colIndex(valuesCol);
    if (iIdx < 0 || cIdx < 0 || vIdx < 0) return this.clone();

    const pivotCols = [...new Set(this.data.map(r => r[cIdx]))].sort();
    const groups = new Map();
    for (const row of this.data) {
      const key = row[iIdx];
      const col = row[cIdx];
      const val = row[vIdx];
      if (!groups.has(key)) groups.set(key, {});
      const g = groups.get(key);
      if (!g[col]) g[col] = [];
      if (typeof val === "number") g[col].push(val);
    }

    const newCols = [indexCol, ...pivotCols.map(String)];
    const newData = [];
    for (const [key, buckets] of groups) {
      const row = [key];
      for (const pc of pivotCols) {
        const vals = buckets[pc] || [];
        switch (aggfunc) {
          case "sum":   row.push(vals.reduce((a, b) => a + b, 0)); break;
          case "mean":  row.push(vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null); break;
          case "count": row.push(vals.length); break;
          case "min":   row.push(vals.length ? Math.min(...vals) : null); break;
          case "max":   row.push(vals.length ? Math.max(...vals) : null); break;
          default:      row.push(vals[0] ?? null);
        }
      }
      newData.push(row);
    }
    return new DataFrame(newCols, newData);
  }

  concat(other) {
    // Align columns – union
    const allCols = [...new Set([...this.columns, ...other.columns])];
    const mapRow = (row, srcCols) => allCols.map(c => {
      const i = srcCols.indexOf(c);
      return i >= 0 ? row[i] : null;
    });
    const data = [
      ...this.data.map(r => mapRow(r, this.columns)),
      ...other.data.map(r => mapRow(r, other.columns)),
    ];
    return new DataFrame(allCols, data, { ...this.roles });
  }

  /* ── statistics ─────────────────────────────────────────────────────── */

  describe() {
    const numCols = this.numericCols();
    const stats = {};
    for (const c of numCols) {
      const vals = this.col(c).filter(v => v != null && typeof v === "number");
      if (!vals.length) continue;
      const sorted = [...vals].sort((a, b) => a - b);
      const n = sorted.length;
      const sum = sorted.reduce((a, b) => a + b, 0);
      const mean = sum / n;
      const variance = sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
      stats[c] = {
        count: n, mean, std: Math.sqrt(variance),
        min: sorted[0], max: sorted[n - 1],
        "25%": sorted[Math.floor(n * 0.25)],
        "50%": sorted[Math.floor(n * 0.5)],
        "75%": sorted[Math.floor(n * 0.75)],
      };
    }
    return stats;
  }

  corr(method = "pearson") {
    const numCols = this.numericCols();
    const arrs = numCols.map(c => this.col(c).map(v => v ?? 0));
    const n = numCols.length;
    const mat = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        row.push(_pearsonR(arrs[i], arrs[j]));
      }
      mat.push(row);
    }
    const data = mat.map((row, i) => [numCols[i], ...row]);
    return new DataFrame(["feature", ...numCols], data);
  }

  valueCounts(col) {
    const vals = this.col(col);
    const counts = {};
    for (const v of vals) { const k = String(v); counts[k] = (counts[k] || 0) + 1; }
    return counts;
  }

  /* ── missing values ─────────────────────────────────────────────────── */

  fillNA(strategy = "mean", constant = 0, subsetCols = null) {
    const df = this.clone();
    const cols = subsetCols || df.columns;
    for (const c of cols) {
      const ci = df.colIndex(c);
      if (ci < 0) continue;
      const vals = df.data.map(r => r[ci]).filter(v => v != null && typeof v === "number");
      let fill;
      switch (strategy) {
        case "mean":   fill = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0; break;
        case "median": {
          const s = [...vals].sort((a, b) => a - b);
          fill = s.length ? s[Math.floor(s.length / 2)] : 0; break;
        }
        case "mode": {
          const freq = {};
          for (const v of df.data.map(r => r[ci]).filter(x => x != null)) freq[v] = (freq[v] || 0) + 1;
          fill = Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
          if (!isNaN(Number(fill))) fill = Number(fill);
          break;
        }
        case "constant": fill = constant; break;
        default: fill = 0;
      }
      for (const row of df.data) {
        if (row[ci] == null) row[ci] = fill;
      }
    }
    return df;
  }

  dropNA(subsetCols = null) {
    const cols = subsetCols || this.columns;
    const idxs = cols.map(c => this.colIndex(c)).filter(i => i >= 0);
    const data = this.data.filter(r => idxs.every(i => r[i] != null));
    return new DataFrame(this.columns, data, { ...this.roles });
  }

  /* ── normalization ──────────────────────────────────────────────────── */

  normalize(method = "z-score", subsetCols = null) {
    const df = this.clone();
    const cols = subsetCols || df.numericCols();
    for (const c of cols) {
      const ci = df.colIndex(c);
      const vals = df.data.map(r => r[ci]).filter(v => v != null && typeof v === "number");
      if (!vals.length) continue;
      if (method === "z-score") {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) || 1;
        for (const row of df.data) if (typeof row[ci] === "number") row[ci] = (row[ci] - mean) / std;
      } else if (method === "min-max") {
        const min = Math.min(...vals), max = Math.max(...vals);
        const range = max - min || 1;
        for (const row of df.data) if (typeof row[ci] === "number") row[ci] = (row[ci] - min) / range;
      } else if (method === "log") {
        for (const row of df.data) if (typeof row[ci] === "number" && row[ci] > 0) row[ci] = Math.log(row[ci]);
      }
    }
    return df;
  }

  /* ── encoding ───────────────────────────────────────────────────────── */

  oneHotEncode(colNames, dropFirst = false) {
    let df = this.clone();
    for (const c of colNames) {
      const ci = df.colIndex(c);
      if (ci < 0) continue;
      const uniq = [...new Set(df.data.map(r => r[ci]).filter(v => v != null))].sort();
      const cats = dropFirst ? uniq.slice(1) : uniq;
      for (const cat of cats) {
        const name = `${c}_${cat}`;
        df.columns.push(name);
        df.data.forEach(r => r.push(r[ci] === cat ? 1 : 0));
      }
      // Remove original
      const newCi = df.colIndex(c);
      df.columns.splice(newCi, 1);
      df.data.forEach(r => r.splice(newCi, 1));
    }
    return df;
  }

  labelEncode(colNames) {
    const df = this.clone();
    const maps = {};
    for (const c of colNames) {
      const ci = df.colIndex(c);
      if (ci < 0) continue;
      const uniq = [...new Set(df.data.map(r => r[ci]).filter(v => v != null))].sort();
      const m = {};
      uniq.forEach((v, i) => { m[v] = i; });
      maps[c] = m;
      for (const row of df.data) if (row[ci] != null) row[ci] = m[row[ci]] ?? row[ci];
    }
    return { df, maps };
  }

  /* ── matrix helpers (for ML) ────────────────────────────────────────── */

  toMatrix(colNames = null) {
    const cols = colNames || this.numericCols();
    const idxs = cols.map(c => this.colIndex(c)).filter(i => i >= 0);
    return this.data.map(r => idxs.map(i => r[i] ?? 0));
  }

  toArray(colName) {
    const idx = this.colIndex(colName);
    return this.data.map(r => r[idx]);
  }
}

/* ── utility functions ───────────────────────────────────────────────────── */

function _seedRng(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

function _pearsonR(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] ?? 0, b = y[i] ?? 0;
    sx += a; sy += b; sxy += a * b; sx2 += a * a; sy2 += b * b;
  }
  const num = n * sxy - sx * sy;
  const den = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
  return den === 0 ? 0 : num / den;
}

function _mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function _std(arr) {
  const m = _mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}
function _unique(arr) { return [...new Set(arr)]; }
