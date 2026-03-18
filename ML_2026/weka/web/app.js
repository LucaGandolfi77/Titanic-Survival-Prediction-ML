/* ================================================================
   PyWeka Web – app.js
   Complete ML engine + UI controller
   Algorithms implemented from scratch in vanilla JS
   ================================================================ */
'use strict';

/* ────────────────────────────────────────────────────────────────
   1. UTILITY FUNCTIONS
   ──────────────────────────────────────────────────────────────── */
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const variance = a => { const m = mean(a); return mean(a.map(x => (x - m) ** 2)); };
const std = a => Math.sqrt(variance(a));
const median = a => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const mode = a => { const f = {}; a.forEach(v => f[v] = (f[v] || 0) + 1); return Object.entries(f).sort((a, b) => b[1] - a[1])[0][0]; };
const unique = a => [...new Set(a)];
const euclidean = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round4 = v => Math.round(v * 1e4) / 1e4;

function seededShuffle(arr, seed = 42) {
  const r = [...arr]; let s = seed;
  const rng = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [r[i], r[j]] = [r[j], r[i]]; }
  return r;
}

function trainTestSplit(X, y, testSize = .2, seed = 42) {
  const idx = seededShuffle([...Array(X.length).keys()], seed);
  const sp = Math.floor(X.length * (1 - testSize));
  return {
    Xtrain: idx.slice(0, sp).map(i => X[i]), ytrain: idx.slice(0, sp).map(i => y[i]),
    Xtest: idx.slice(sp).map(i => X[i]),   ytest: idx.slice(sp).map(i => y[i]),
  };
}

function kFoldSplit(n, k = 5, seed = 42) {
  const idx = seededShuffle([...Array(n).keys()], seed), sz = Math.floor(n / k), folds = [];
  for (let i = 0; i < k; i++) {
    const s = i * sz, e = i === k - 1 ? n : s + sz;
    folds.push({ test: idx.slice(s, e), train: [...idx.slice(0, s), ...idx.slice(e)] });
  }
  return folds;
}

function scaleData(X) {
  const p = X[0].length, mu = [], sig = [];
  for (let j = 0; j < p; j++) { const c = X.map(r => r[j]); mu.push(mean(c)); sig.push(std(c) || 1); }
  return { data: X.map(r => r.map((v, j) => (v - mu[j]) / sig[j])), mu, sig };
}

/* ────────────────────────────────────────────────────────────────
   2. EVALUATION METRICS
   ──────────────────────────────────────────────────────────────── */
function accuracy(yt, yp) { let c = 0; for (let i = 0; i < yt.length; i++) if (yt[i] === yp[i]) c++; return c / yt.length; }

function confusionMatrix(yt, yp) {
  const cls = unique([...yt, ...yp]).sort((a, b) => a - b), n = cls.length;
  const m = Array.from({ length: n }, () => Array(n).fill(0)), idx = {};
  cls.forEach((c, i) => idx[c] = i);
  for (let i = 0; i < yt.length; i++) m[idx[yt[i]]][idx[yp[i]]]++;
  return { matrix: m, classes: cls };
}

function weightedF1(yt, yp) {
  const cls = unique([...yt, ...yp]).sort((a, b) => a - b);
  let wf1 = 0, wp = 0, wr = 0, total = yt.length;
  for (const c of cls) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yt.length; i++) {
      if (yp[i] === c && yt[i] === c) tp++;
      else if (yp[i] === c) fp++;
      else if (yt[i] === c) fn++;
    }
    const sup = tp + fn, p = tp + fp > 0 ? tp / (tp + fp) : 0, r = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f = p + r > 0 ? 2 * p * r / (p + r) : 0;
    const w = sup / total;
    wf1 += f * w; wp += p * w; wr += r * w;
  }
  return { precision: wp, recall: wr, f1: wf1 };
}

function r2Score(yt, yp) { const m = mean(yt); let sr = 0, st = 0; for (let i = 0; i < yt.length; i++) { sr += (yt[i] - yp[i]) ** 2; st += (yt[i] - m) ** 2; } return st > 0 ? 1 - sr / st : 0; }
function rmseScore(yt, yp) { let s = 0; for (let i = 0; i < yt.length; i++) s += (yt[i] - yp[i]) ** 2; return Math.sqrt(s / yt.length); }
function maeScore(yt, yp) { let s = 0; for (let i = 0; i < yt.length; i++) s += Math.abs(yt[i] - yp[i]); return s / yt.length; }

function silhouetteScore(X, labels) {
  const n = Math.min(X.length, 800);
  const idx = seededShuffle([...Array(X.length).keys()], 42).slice(0, n);
  const Xs = idx.map(i => X[i]), ls = idx.map(i => labels[i]);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const same = [], other = {};
    for (let j = 0; j < n; j++) { if (i === j) continue; const d = euclidean(Xs[i], Xs[j]); if (ls[j] === ls[i]) same.push(d); else { (other[ls[j]] = other[ls[j]] || []).push(d); } }
    const a = same.length > 0 ? mean(same) : 0;
    const bVals = Object.values(other).map(ds => mean(ds));
    const b = bVals.length > 0 ? Math.min(...bVals) : 0;
    total += Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
  }
  return total / n;
}

/* ────────────────────────────────────────────────────────────────
   3. ML ALGORITHMS – CLASSIFICATION
   ──────────────────────────────────────────────────────────────── */

class KNNClassifier {
  constructor(k = 5) { this.k = k; }
  fit(X, y) { this.X = X; this.y = y; }
  predict(Xp) {
    return Xp.map(x => {
      const ds = this.X.map((xi, i) => [euclidean(x, xi), this.y[i]]).sort((a, b) => a[0] - b[0]).slice(0, this.k);
      const v = {}; ds.forEach(([, l]) => v[l] = (v[l] || 0) + 1);
      return +Object.entries(v).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

class GaussianNB {
  fit(X, y) {
    this.cls = unique(y).sort((a, b) => a - b);
    this.stats = {}; this.prior = {};
    for (const c of this.cls) {
      const rows = X.filter((_, i) => y[i] === c);
      this.prior[c] = rows.length / y.length;
      this.stats[c] = X[0].map((_, j) => { const col = rows.map(r => r[j]); return { m: mean(col), s: std(col) + 1e-9 }; });
    }
  }
  predict(Xp) {
    return Xp.map(x => {
      let best = this.cls[0], bestS = -Infinity;
      for (const c of this.cls) {
        let lp = Math.log(this.prior[c]);
        for (let j = 0; j < x.length; j++) { const { m, s } = this.stats[c][j]; lp += -.5 * Math.log(2 * Math.PI * s * s) - (x[j] - m) ** 2 / (2 * s * s); }
        if (lp > bestS) { bestS = lp; best = c; }
      }
      return best;
    });
  }
}

class DecisionTreeClassifier {
  constructor(maxDepth = 10, minLeaf = 5) { this.maxDepth = maxDepth; this.minLeaf = minLeaf; }
  fit(X, y) { this.nFeat = X[0].length; this.root = this._build(X, y, 0); }
  predict(Xp) { return Xp.map(x => this._pred(x, this.root)); }

  getImportances() {
    const imp = Array(this.nFeat).fill(0);
    this._accImp(this.root, imp);
    const t = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / t);
  }

  _build(X, y, d) {
    if (d >= this.maxDepth || y.length <= this.minLeaf || unique(y).length === 1)
      return { leaf: true, val: this._majority(y), n: y.length };
    const best = this._bestSplit(X, y);
    if (best.gain <= 0 || !best.li.length || !best.ri.length)
      return { leaf: true, val: this._majority(y), n: y.length };
    return {
      leaf: false, f: best.f, t: best.t, gain: best.gain, n: y.length,
      l: this._build(best.li.map(i => X[i]), best.li.map(i => y[i]), d + 1),
      r: this._build(best.ri.map(i => X[i]), best.ri.map(i => y[i]), d + 1),
    };
  }

  _bestSplit(X, y) {
    let best = { gain: -Infinity, f: 0, t: 0, li: [], ri: [] };
    const pg = this._gini(y), n = y.length;
    for (let f = 0; f < X[0].length; f++) {
      const uv = unique(X.map(r => r[f]).filter(v => !isNaN(v))).sort((a, b) => a - b);
      const thresholds = uv.length <= 20
        ? uv.slice(0, -1).map((v, i) => (v + uv[i + 1]) / 2)
        : Array.from({ length: 19 }, (_, q) => uv[Math.floor((q + 1) * uv.length / 20)]);
      for (const t of unique(thresholds)) {
        const li = [], ri = [];
        for (let i = 0; i < n; i++) (X[i][f] <= t ? li : ri).push(i);
        if (!li.length || !ri.length) continue;
        const g = pg - (li.length / n) * this._gini(li.map(i => y[i])) - (ri.length / n) * this._gini(ri.map(i => y[i]));
        if (g > best.gain) best = { gain: g, f, t, li, ri };
      }
    }
    return best;
  }

  _gini(y) { const c = {}; y.forEach(v => c[v] = (c[v] || 0) + 1); let g = 1; for (const n of Object.values(c)) g -= (n / y.length) ** 2; return g; }
  _majority(y) { const c = {}; y.forEach(v => c[v] = (c[v] || 0) + 1); return +Object.entries(c).sort((a, b) => b[1] - a[1])[0][0]; }
  _pred(x, nd) { return nd.leaf ? nd.val : (x[nd.f] <= nd.t ? this._pred(x, nd.l) : this._pred(x, nd.r)); }
  _accImp(nd, imp) { if (!nd.leaf) { imp[nd.f] += (nd.gain || 0) * nd.n; this._accImp(nd.l, imp); this._accImp(nd.r, imp); } }
}

class LogisticRegressionClassifier {
  constructor(lr = .01, maxIter = 400) { this.lr = lr; this.maxIter = maxIter; }
  fit(X, y) {
    this.cls = unique(y).sort((a, b) => a - b);
    const n = X.length, p = X[0].length;
    if (this.cls.length === 2) {
      this.w = Array(p).fill(0); this.b = 0;
      for (let it = 0; it < this.maxIter; it++)
        for (let i = 0; i < n; i++) {
          const z = clamp(X[i].reduce((s, v, j) => s + v * this.w[j], 0) + this.b, -500, 500);
          const e = 1 / (1 + Math.exp(-z)) - (y[i] === this.cls[1] ? 1 : 0);
          for (let j = 0; j < p; j++) this.w[j] -= this.lr * e * X[i][j];
          this.b -= this.lr * e;
        }
    } else {
      this.models = {};
      for (const c of this.cls) {
        const w = Array(p).fill(0); let b = 0;
        for (let it = 0; it < this.maxIter; it++)
          for (let i = 0; i < n; i++) {
            const z = clamp(X[i].reduce((s, v, j) => s + v * w[j], 0) + b, -500, 500);
            const e = 1 / (1 + Math.exp(-z)) - (y[i] === c ? 1 : 0);
            for (let j = 0; j < p; j++) w[j] -= this.lr * e * X[i][j];
            b -= this.lr * e;
          }
        this.models[c] = { w, b };
      }
    }
  }
  predict(Xp) {
    if (this.cls.length === 2) {
      return Xp.map(x => {
        const z = clamp(x.reduce((s, v, j) => s + v * this.w[j], 0) + this.b, -500, 500);
        return 1 / (1 + Math.exp(-z)) >= .5 ? this.cls[1] : this.cls[0];
      });
    }
    return Xp.map(x => {
      let best = this.cls[0], bs = -Infinity;
      for (const c of this.cls) { const { w, b } = this.models[c]; const z = x.reduce((s, v, j) => s + v * w[j], 0) + b; if (z > bs) { bs = z; best = c; } }
      return best;
    });
  }
}

class RandomForestClassifier {
  constructor(nTrees = 12, maxDepth = 8) { this.nTrees = nTrees; this.maxDepth = maxDepth; }
  fit(X, y) {
    const n = X.length, p = X[0].length, mf = Math.max(1, Math.floor(Math.sqrt(p)));
    this.trees = []; this.feats = [];
    for (let t = 0; t < this.nTrees; t++) {
      const bi = Array.from({ length: n }, () => Math.floor(Math.random() * n));
      const fs = seededShuffle([...Array(p).keys()], t * 7 + 13).slice(0, mf);
      this.feats.push(fs);
      const bX = bi.map(i => fs.map(f => X[i][f])), bY = bi.map(i => y[i]);
      const tree = new DecisionTreeClassifier(this.maxDepth, 5);
      tree.fit(bX, bY); this.trees.push(tree);
    }
  }
  predict(Xp) {
    return Xp.map(x => {
      const v = {};
      for (let t = 0; t < this.trees.length; t++) { const px = this.feats[t].map(f => x[f]); const pr = this.trees[t].predict([px])[0]; v[pr] = (v[pr] || 0) + 1; }
      return +Object.entries(v).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

/* ────────────────────────────────────────────────────────────────
   4. ML ALGORITHMS – REGRESSION
   ──────────────────────────────────────────────────────────────── */

class LinearRegressionModel {
  fit(X, y) {
    const n = X.length, p = X[0].length; this.w = Array(p).fill(0); this.b = 0;
    for (let it = 0; it < 800; it++) {
      const gw = Array(p).fill(0); let gb = 0;
      for (let i = 0; i < n; i++) {
        let pr = this.b; for (let j = 0; j < p; j++) pr += this.w[j] * X[i][j];
        const e = pr - y[i]; for (let j = 0; j < p; j++) gw[j] += e * X[i][j]; gb += e;
      }
      const lr = .01; for (let j = 0; j < p; j++) this.w[j] -= lr * gw[j] / n; this.b -= lr * gb / n;
    }
  }
  predict(Xp) { return Xp.map(x => { let p = this.b; for (let j = 0; j < x.length; j++) p += this.w[j] * x[j]; return p; }); }
}

class RidgeRegressionModel {
  constructor(alpha = 1) { this.alpha = alpha; }
  fit(X, y) {
    const n = X.length, p = X[0].length; this.w = Array(p).fill(0); this.b = 0;
    for (let it = 0; it < 800; it++) {
      const gw = Array(p).fill(0); let gb = 0;
      for (let i = 0; i < n; i++) {
        let pr = this.b; for (let j = 0; j < p; j++) pr += this.w[j] * X[i][j];
        const e = pr - y[i]; for (let j = 0; j < p; j++) gw[j] += e * X[i][j]; gb += e;
      }
      const lr = .01; for (let j = 0; j < p; j++) this.w[j] -= lr * (gw[j] / n + this.alpha * this.w[j]); this.b -= lr * gb / n;
    }
  }
  predict(Xp) { return Xp.map(x => { let p = this.b; for (let j = 0; j < x.length; j++) p += this.w[j] * x[j]; return p; }); }
}

class KNNRegressor {
  constructor(k = 5) { this.k = k; }
  fit(X, y) { this.X = X; this.y = y; }
  predict(Xp) {
    return Xp.map(x => {
      const ds = this.X.map((xi, i) => [euclidean(x, xi), this.y[i]]).sort((a, b) => a[0] - b[0]).slice(0, this.k);
      return mean(ds.map(d => d[1]));
    });
  }
}

class DecisionTreeRegressor {
  constructor(maxDepth = 10, minLeaf = 5) { this.maxDepth = maxDepth; this.minLeaf = minLeaf; }
  fit(X, y) { this.root = this._build(X, y, 0); }
  predict(Xp) { return Xp.map(x => this._pred(x, this.root)); }

  _build(X, y, d) {
    if (d >= this.maxDepth || y.length <= this.minLeaf || unique(y).length === 1)
      return { leaf: true, val: mean(y) };
    const best = this._bestSplit(X, y);
    if (best.gain <= 0 || !best.li.length || !best.ri.length) return { leaf: true, val: mean(y) };
    return {
      leaf: false, f: best.f, t: best.t,
      l: this._build(best.li.map(i => X[i]), best.li.map(i => y[i]), d + 1),
      r: this._build(best.ri.map(i => X[i]), best.ri.map(i => y[i]), d + 1),
    };
  }

  _bestSplit(X, y) {
    let best = { gain: -Infinity, f: 0, t: 0, li: [], ri: [] };
    const pv = variance(y) * y.length, n = y.length;
    for (let f = 0; f < X[0].length; f++) {
      const uv = unique(X.map(r => r[f]).filter(v => !isNaN(v))).sort((a, b) => a - b);
      const thresholds = uv.length <= 20
        ? uv.slice(0, -1).map((v, i) => (v + uv[i + 1]) / 2)
        : Array.from({ length: 19 }, (_, q) => uv[Math.floor((q + 1) * uv.length / 20)]);
      for (const t of unique(thresholds)) {
        const li = [], ri = [];
        for (let i = 0; i < n; i++) (X[i][f] <= t ? li : ri).push(i);
        if (!li.length || !ri.length) continue;
        const g = pv - variance(li.map(i => y[i])) * li.length - variance(ri.map(i => y[i])) * ri.length;
        if (g > best.gain) best = { gain: g, f, t, li, ri };
      }
    }
    return best;
  }

  _pred(x, nd) { return nd.leaf ? nd.val : (x[nd.f] <= nd.t ? this._pred(x, nd.l) : this._pred(x, nd.r)); }
}

/* ────────────────────────────────────────────────────────────────
   5. ML ALGORITHMS – CLUSTERING
   ──────────────────────────────────────────────────────────────── */

class KMeansModel {
  constructor(k = 3, maxIter = 100) { this.k = k; this.maxIter = maxIter; }
  fit(X) {
    const n = X.length, d = X[0].length;
    // K-Means++ init
    this.centroids = [X[Math.floor(Math.random() * n)]];
    while (this.centroids.length < this.k) {
      const ds = X.map(x => Math.min(...this.centroids.map(c => euclidean(x, c))));
      const total = ds.reduce((a, b) => a + b, 0);
      let r = Math.random() * total;
      for (let i = 0; i < n; i++) { r -= ds[i]; if (r <= 0) { this.centroids.push([...X[i]]); break; } }
    }
    this.labels = Array(n).fill(0);
    for (let it = 0; it < this.maxIter; it++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let bk = 0, bd = Infinity;
        for (let k = 0; k < this.k; k++) { const dd = euclidean(X[i], this.centroids[k]); if (dd < bd) { bd = dd; bk = k; } }
        if (this.labels[i] !== bk) changed = true;
        this.labels[i] = bk;
      }
      if (!changed) break;
      for (let k = 0; k < this.k; k++) {
        const mem = X.filter((_, i) => this.labels[i] === k);
        if (!mem.length) continue;
        this.centroids[k] = Array(d).fill(0);
        for (const m of mem) for (let j = 0; j < d; j++) this.centroids[k][j] += m[j];
        for (let j = 0; j < d; j++) this.centroids[k][j] /= mem.length;
      }
    }
    this.inertia = 0;
    for (let i = 0; i < n; i++) this.inertia += euclidean(X[i], this.centroids[this.labels[i]]) ** 2;
    return this.labels;
  }
}

/* ────────────────────────────────────────────────────────────────
   6. SAMPLE DATASETS (synthetic, seeded)
   ──────────────────────────────────────────────────────────────── */

function generateSample(name) {
  let s = 42;
  const rng = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const rnorm = () => { let u = 0, v = 0; while (!u) u = rng(); v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  if (name === 'iris') {
    const cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'];
    const specs = [
      { c: 'setosa', m: [5.0, 3.4, 1.5, 0.2], s: [.35, .38, .17, .11] },
      { c: 'versicolor', m: [5.9, 2.8, 4.3, 1.3], s: [.52, .31, .47, .20] },
      { c: 'virginica', m: [6.6, 3.0, 5.6, 2.0], s: [.64, .32, .55, .27] },
    ];
    const data = [];
    for (const sp of specs) for (let i = 0; i < 50; i++) {
      const row = {};
      for (let j = 0; j < 4; j++) row[cols[j]] = round4(sp.m[j] + rnorm() * sp.s[j]);
      row.species = sp.c;
      data.push(row);
    }
    return { columns: cols, data };
  }

  if (name === 'wine') {
    const cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid', 'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline', 'class'];
    const specs = [
      { c: 'class_1', m: [13.7, 2.0, 2.5, 17.0, 106, 2.8, 3.0, 0.29, 1.9, 5.5, 1.06, 3.2, 1100], s: [.5, .6, .2, 2.5, 12, .3, .4, .07, .4, 1.1, .12, .3, 200] },
      { c: 'class_2', m: [12.5, 1.9, 2.2, 20.0, 94, 2.3, 2.0, 0.36, 1.6, 3.1, 1.06, 2.8, 520], s: [.5, .8, .3, 3.0, 15, .4, .6, .12, .5, 1.0, .2, .4, 160] },
      { c: 'class_3', m: [13.1, 3.3, 2.4, 21.5, 99, 1.7, 0.8, 0.45, 1.2, 7.4, 0.68, 1.7, 630], s: [.5, .9, .2, 3.5, 12, .3, .3, .1, .3, 2.0, .1, .3, 170] },
    ];
    const data = [];
    for (const sp of specs) for (let i = 0; i < 59; i++) {
      const row = {};
      for (let j = 0; j < 13; j++) row[cols[j]] = round4(Math.max(0, sp.m[j] + rnorm() * sp.s[j]));
      row['class'] = sp.c;
      data.push(row);
    }
    return { columns: cols, data };
  }

  if (name === 'diabetes') {
    const cols = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target'];
    const data = [];
    for (let i = 0; i < 200; i++) {
      const row = {};
      row.age = round4(rnorm() * .05);
      row.bmi = round4(rnorm() * .05);
      row.bp = round4(rnorm() * .05);
      for (let j = 1; j <= 6; j++) row['s' + j] = round4(rnorm() * .05);
      row.target = round4(150 + row.age * 400 + row.bmi * 600 + row.bp * 300 + rnorm() * 50);
      data.push(row);
    }
    return { columns: cols, data };
  }

  return null;
}

/* ────────────────────────────────────────────────────────────────
   7. PLOTLY DARK LAYOUT HELPER
   ──────────────────────────────────────────────────────────────── */

const DARK_LAYOUT = {
  paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b',
  font: { color: '#e2e8f0', family: 'Inter, sans-serif', size: 12 },
  xaxis: { gridcolor: '#334155', zerolinecolor: '#475569' },
  yaxis: { gridcolor: '#334155', zerolinecolor: '#475569' },
  margin: { t: 40, r: 20, b: 50, l: 55 },
  colorway: ['#a855f7', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#06b6d4', '#f97316'],
};

function darkLayout(overrides = {}) {
  return JSON.parse(JSON.stringify({ ...DARK_LAYOUT, ...overrides }));
}

/* ────────────────────────────────────────────────────────────────
   8. APP CONTROLLER
   ──────────────────────────────────────────────────────────────── */

class PyWekaApp {
  constructor() {
    this.data = null;
    this.columns = [];
    this.types = {};
    this.history = [];

    this.classifierDefs = {
      'K-Nearest Neighbors': () => new KNNClassifier(5),
      'Gaussian Naive Bayes': () => new GaussianNB(),
      'Decision Tree': () => new DecisionTreeClassifier(10, 5),
      'Logistic Regression': () => new LogisticRegressionClassifier(.01, 400),
      'Random Forest': () => new RandomForestClassifier(12, 8),
    };
    this.regressorDefs = {
      'Linear Regression': () => new LinearRegressionModel(),
      'Ridge Regression': () => new RidgeRegressionModel(1),
      'KNN Regressor': () => new KNNRegressor(5),
      'Decision Tree': () => new DecisionTreeRegressor(10, 5),
    };

    this._initUI();
  }

  /* ── UI Initialization ───────────────────────────────────── */
  _initUI() {
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      b.classList.add('active');
      document.getElementById('tab-' + b.dataset.tab).classList.add('active');
    }));

    // File upload
    const dz = document.getElementById('drop-zone'), fi = document.getElementById('file-input');
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag-over'); if (e.dataTransfer.files[0]) this._loadFile(e.dataTransfer.files[0]); });
    dz.addEventListener('click', () => fi.click());
    fi.addEventListener('change', e => { if (e.target.files[0]) this._loadFile(e.target.files[0]); });

    // Populate algo lists
    this._populateAlgos('clf-algo-list', Object.keys(this.classifierDefs));
    this._populateAlgos('reg-algo-list', Object.keys(this.regressorDefs));
  }

  _populateAlgos(containerId, names) {
    const el = document.getElementById(containerId);
    el.innerHTML = names.map(n => `<label class="algo-check"><input type="checkbox" checked data-algo="${n}"><span>${n}</span></label>`).join('');
  }

  /* ── Data Loading ────────────────────────────────────────── */
  _loadFile(file) {
    this.setStatus('Loading ' + file.name + '...');
    Papa.parse(file, {
      header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: r => { this._setData(r.data, r.meta.fields, file.name); },
      error: e => alert('Parse error: ' + e.message),
    });
  }

  loadSample(name) {
    this.setStatus('Generating ' + name + ' sample...');
    const s = generateSample(name);
    if (s) this._setData(s.data, s.columns, name + '_dataset');
  }

  _setData(data, columns, name) {
    // Clean rows: remove empty last row
    this.data = data.filter(r => columns.some(c => r[c] != null && r[c] !== ''));
    this.columns = columns;
    this._detectTypes();
    this.history = [JSON.parse(JSON.stringify(this.data))];
    this._refreshAll();
    this.setStatus(`Loaded: ${name} (${this.data.length} rows x ${this.columns.length} cols)`);
    document.getElementById('dataset-info').textContent = `${name} (${this.data.length} x ${this.columns.length})`;
  }

  _detectTypes() {
    this.types = {};
    for (const c of this.columns) {
      const vals = this.data.map(r => r[c]).filter(v => v != null && v !== '');
      this.types[c] = vals.length > 0 && vals.every(v => typeof v === 'number' && !isNaN(v)) ? 'numeric' : 'categorical';
    }
  }

  /* ── Full UI Refresh ─────────────────────────────────────── */
  _refreshAll() {
    if (!this.data) return;
    document.getElementById('drop-zone').classList.add('hidden');
    document.getElementById('preprocess-content').classList.remove('hidden');

    const numCols = this.columns.filter(c => this.types[c] === 'numeric');
    const catCols = this.columns.filter(c => this.types[c] === 'categorical');
    let miss = 0;
    for (const r of this.data) for (const c of this.columns) if (r[c] == null || r[c] === '' || (typeof r[c] === 'number' && isNaN(r[c]))) miss++;

    document.getElementById('sum-rows').textContent = this.data.length;
    document.getElementById('sum-cols').textContent = this.columns.length;
    document.getElementById('sum-numeric').textContent = numCols.length;
    document.getElementById('sum-cat').textContent = catCols.length;
    document.getElementById('sum-missing').textContent = miss;

    // Attribute list
    const al = document.getElementById('attr-list');
    al.innerHTML = this.columns.map(c => {
      const t = this.types[c], m = this.data.filter(r => r[c] == null || r[c] === '' || (typeof r[c] === 'number' && isNaN(r[c]))).length;
      return `<div class="attr-item" onclick="app.showAttrDetails('${c.replace(/'/g, "\\'")}')">
        <span class="attr-name">${c}</span>
        <span class="attr-badge ${t}">${t === 'numeric' ? 'NUM' : 'CAT'}</span>
        ${m > 0 ? `<span class="attr-miss">${m} NaN</span>` : ''}
      </div>`;
    }).join('');

    // Operations column selector
    const opsCol = document.getElementById('ops-column');
    opsCol.innerHTML = this.columns.map(c => `<option value="${c}">${c}</option>`).join('');

    // Data table
    this._renderTable();

    // Populate selectors on other tabs
    this._populateSelects();

    // Cluster features
    const cf = document.getElementById('clu-feat-list');
    cf.innerHTML = numCols.map(c => `<label class="algo-check"><input type="checkbox" checked data-feat="${c}"><span>${c}</span></label>`).join('');
  }

  _renderTable() {
    const t = document.getElementById('data-table');
    const cols = this.columns.slice(0, 25);
    t.querySelector('thead').innerHTML = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
    t.querySelector('tbody').innerHTML = this.data.slice(0, 100).map(r =>
      '<tr>' + cols.map(c => { const v = r[c]; return `<td>${v == null || (typeof v === 'number' && isNaN(v)) ? '<em style="color:var(--err)">?</em>' : String(v).slice(0, 25)}</td>`; }).join('') + '</tr>'
    ).join('');
  }

  _populateSelects() {
    const all = this.columns, num = this.columns.filter(c => this.types[c] === 'numeric');
    const opts = cs => cs.map(c => `<option value="${c}">${c}</option>`).join('');

    document.getElementById('clf-target').innerHTML = opts(all);
    if (all.length) document.getElementById('clf-target').value = all[all.length - 1];
    document.getElementById('reg-target').innerHTML = opts(num);
    if (num.length) document.getElementById('reg-target').value = num[num.length - 1];
    document.getElementById('sel-target').innerHTML = opts(all);
    if (all.length) document.getElementById('sel-target').value = all[all.length - 1];

    // Viz
    document.getElementById('viz-x').innerHTML = opts(all);
    document.getElementById('viz-y').innerHTML = opts(all);
    if (all.length > 1) document.getElementById('viz-y').value = all[1];
    document.getElementById('viz-hue').innerHTML = '<option value="">(none)</option>' + opts(all);
  }

  showAttrDetails(col) {
    document.querySelectorAll('.attr-item').forEach(el => el.classList.remove('active'));
    // Find and activate
    document.querySelectorAll('.attr-item').forEach(el => { if (el.querySelector('.attr-name').textContent === col) el.classList.add('active'); });

    const el = document.getElementById('attr-details');
    const vals = this.data.map(r => r[col]).filter(v => v != null && v !== '' && !(typeof v === 'number' && isNaN(v)));
    const miss = this.data.length - vals.length;
    let html = `<b>${col}</b>\nType: ${this.types[col]}\nCount: ${vals.length}\nMissing: ${miss}\nUnique: ${unique(vals).length}\n`;

    if (this.types[col] === 'numeric') {
      const nums = vals.map(Number).filter(v => !isNaN(v));
      if (nums.length) {
        const s = [...nums].sort((a, b) => a - b);
        html += `\nMean: ${round4(mean(nums))}\nStd: ${round4(std(nums))}\nMin: ${s[0]}\n25%: ${s[Math.floor(s.length * .25)]}\n50%: ${median(nums)}\n75%: ${s[Math.floor(s.length * .75)]}\nMax: ${s[s.length - 1]}`;
      }
    } else {
      const freq = {}; vals.forEach(v => freq[v] = (freq[v] || 0) + 1);
      const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 10);
      html += '\nTop values:';
      for (const [v, c] of sorted) html += `\n  ${v}: ${c}`;
    }
    el.textContent = html;
    document.getElementById('ops-column').value = col;
  }

  /* ── Preprocessing Operations ────────────────────────────── */
  _getCol() { return document.getElementById('ops-column').value; }

  _checkpoint() { this.history.push(JSON.parse(JSON.stringify(this.data))); }

  undo() {
    if (this.history.length > 1) {
      this.history.pop();
      this.data = JSON.parse(JSON.stringify(this.history[this.history.length - 1]));
      this._detectTypes();
      this._refreshAll();
      this.setStatus('Undo successful');
    } else { this.setStatus('Nothing to undo'); }
  }

  fillMissing(strategy) {
    const col = this._getCol(); if (!col) return;
    this._checkpoint();
    if (strategy === 'drop') {
      this.data = this.data.filter(r => r[col] != null && r[col] !== '' && !(typeof r[col] === 'number' && isNaN(r[col])));
    } else {
      const vals = this.data.map(r => r[col]).filter(v => v != null && v !== '' && !(typeof v === 'number' && isNaN(v)));
      let fill;
      if (strategy === 'mean' && this.types[col] === 'numeric') fill = mean(vals.map(Number));
      else if (strategy === 'median' && this.types[col] === 'numeric') fill = median(vals.map(Number));
      else if (strategy === 'mode') fill = mode(vals.map(String));
      else if (strategy === 'zero') fill = 0;
      else fill = 0;
      this.data.forEach(r => { if (r[col] == null || r[col] === '' || (typeof r[col] === 'number' && isNaN(r[col]))) r[col] = fill; });
    }
    this._detectTypes(); this._refreshAll();
    this.setStatus(`Filled missing in '${col}' with ${strategy}`);
  }

  dropAllMissing() {
    this._checkpoint();
    this.data = this.data.filter(r => this.columns.every(c => r[c] != null && r[c] !== '' && !(typeof r[c] === 'number' && isNaN(r[c]))));
    this._refreshAll(); this.setStatus('Dropped all rows with missing values');
  }

  labelEncode() {
    const col = this._getCol(); if (!col) return;
    this._checkpoint();
    const vals = unique(this.data.map(r => String(r[col] ?? '')));
    const map = {}; vals.sort().forEach((v, i) => map[v] = i);
    this.data.forEach(r => r[col] = map[String(r[col] ?? '')]);
    this._detectTypes(); this._refreshAll();
    this.setStatus(`Label-encoded '${col}' (${vals.length} classes)`);
  }

  oneHotEncode() {
    const col = this._getCol(); if (!col) return;
    this._checkpoint();
    const vals = unique(this.data.map(r => String(r[col] ?? ''))).sort();
    if (vals.length > 20) { alert('Too many unique values (>' + 20 + '). Use label encoding.'); return; }
    this.data = this.data.map(r => {
      const nr = { ...r };
      for (const v of vals) nr[col + '_' + v] = (String(r[col]) === v ? 1 : 0);
      delete nr[col];
      return nr;
    });
    this.columns = Object.keys(this.data[0]);
    this._detectTypes(); this._refreshAll();
    this.setStatus(`One-hot encoded '${col}'`);
  }

  standardize() {
    if (!this.data) return;
    this._checkpoint();
    const numCols = this.columns.filter(c => this.types[c] === 'numeric');
    for (const c of numCols) {
      const vals = this.data.map(r => r[c]).filter(v => typeof v === 'number' && !isNaN(v));
      const m = mean(vals), s = std(vals) || 1;
      this.data.forEach(r => { if (typeof r[c] === 'number') r[c] = round4((r[c] - m) / s); });
    }
    this._refreshAll(); this.setStatus(`Standardized ${numCols.length} numeric column(s)`);
  }

  normalize() {
    if (!this.data) return;
    this._checkpoint();
    const numCols = this.columns.filter(c => this.types[c] === 'numeric');
    for (const c of numCols) {
      const vals = this.data.map(r => r[c]).filter(v => typeof v === 'number' && !isNaN(v));
      const mn = Math.min(...vals), mx = Math.max(...vals), rng = mx - mn || 1;
      this.data.forEach(r => { if (typeof r[c] === 'number') r[c] = round4((r[c] - mn) / rng); });
    }
    this._refreshAll(); this.setStatus(`Normalized ${numCols.length} numeric column(s)`);
  }

  removeColumn() {
    const col = this._getCol(); if (!col) return;
    this._checkpoint();
    this.data.forEach(r => delete r[col]);
    this.columns = this.columns.filter(c => c !== col);
    this._detectTypes(); this._refreshAll();
    this.setStatus(`Removed column '${col}'`);
  }

  removeOutliers() {
    const col = this._getCol(); if (!col || this.types[col] !== 'numeric') { alert('Select a numeric column.'); return; }
    this._checkpoint();
    const vals = this.data.map(r => r[col]).filter(v => typeof v === 'number' && !isNaN(v));
    const q1 = vals.sort((a, b) => a - b)[Math.floor(vals.length * .25)];
    const q3 = vals[Math.floor(vals.length * .75)];
    const iqr = q3 - q1, lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
    const before = this.data.length;
    this.data = this.data.filter(r => typeof r[col] !== 'number' || (r[col] >= lo && r[col] <= hi));
    this._refreshAll(); this.setStatus(`Removed ${before - this.data.length} outliers from '${col}'`);
  }

  /* ── Prepare data for ML ─────────────────────────────────── */
  _prepareData(targetCol, task = 'classification') {
    const featCols = this.columns.filter(c => c !== targetCol && this.types[c] === 'numeric');
    if (!featCols.length) throw new Error('No numeric features. Encode categoricals first.');
    const valid = this.data.filter(r =>
      r[targetCol] != null && r[targetCol] !== '' && !(typeof r[targetCol] === 'number' && isNaN(r[targetCol])) &&
      featCols.every(c => typeof r[c] === 'number' && !isNaN(r[c]))
    );
    if (valid.length < 10) throw new Error('Not enough valid rows (< 10) after filtering NaNs.');
    const X = valid.map(r => featCols.map(c => r[c]));
    let y, classLabels = null;
    if (task === 'classification') {
      const tv = valid.map(r => String(r[targetCol]));
      classLabels = unique(tv).sort();
      const lm = {}; classLabels.forEach((l, i) => lm[l] = i);
      y = tv.map(v => lm[v]);
    } else {
      y = valid.map(r => +r[targetCol]);
      if (y.some(v => isNaN(v))) throw new Error('Target must be numeric for regression.');
    }
    return { X, y, featCols, classLabels };
  }

  /* ── Classification ──────────────────────────────────────── */
  runClassify() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const target = document.getElementById('clf-target').value;
    const selected = [...document.querySelectorAll('#clf-algo-list input:checked')].map(el => el.dataset.algo);
    if (!selected.length) { alert('Select at least one algorithm.'); return; }
    const evalMode = document.querySelector('input[name="clf-eval"]:checked').value;
    const testPct = +document.getElementById('clf-test').value / 100;
    const folds = +document.getElementById('clf-folds').value;
    const doScale = document.getElementById('clf-scale').checked;

    let prep;
    try { prep = this._prepareData(target, 'classification'); } catch (e) { alert(e.message); return; }
    let { X, y, featCols, classLabels } = prep;

    // Subsample for browser performance
    if (X.length > 5000) {
      const idx = seededShuffle([...Array(X.length).keys()], 42).slice(0, 5000);
      X = idx.map(i => X[i]); y = idx.map(i => y[i]);
    }

    const box = document.getElementById('clf-results');
    box.innerHTML = '';
    const log = t => box.innerHTML += t + '\n';
    log(`<span class="r-head">Classification  |  Target: ${target}  |  Mode: ${evalMode}</span>`);
    log(`<span class="r-head">Instances: ${X.length}  |  Features: ${featCols.length}  |  Classes: ${classLabels.length} [${classLabels.join(', ')}]</span>`);
    log(`<span class="r-sep">${'='.repeat(72)}</span>`);
    this.setStatus('Running classifiers...');

    const results = [];
    setTimeout(() => {
      for (const name of selected) {
        const factory = this.classifierDefs[name]; if (!factory) continue;
        try {
          const model = factory();
          let yt, yp, t0, elapsed;

          if (evalMode === 'cv') {
            const cvFolds = kFoldSplit(X.length, folds, 42);
            const allPred = Array(X.length);
            t0 = performance.now();
            for (const { train, test } of cvFolds) {
              let Xtr = train.map(i => X[i]), ytr = train.map(i => y[i]);
              let Xte = test.map(i => X[i]);
              if (doScale) { const s = scaleData(Xtr); Xtr = s.data; Xte = Xte.map(r => r.map((v, j) => (v - s.mu[j]) / s.sig[j])); }
              model.fit(Xtr, ytr);
              const pr = model.predict(Xte);
              test.forEach((idx, i) => allPred[idx] = pr[i]);
            }
            elapsed = performance.now() - t0;
            yt = y; yp = allPred;
          } else if (evalMode === 'train') {
            let Xs = X;
            if (doScale) Xs = scaleData(X).data;
            t0 = performance.now(); model.fit(Xs, y); yp = model.predict(Xs); elapsed = performance.now() - t0;
            yt = y;
          } else {
            const sp = trainTestSplit(X, y, testPct, 42);
            let Xtr = sp.Xtrain, Xte = sp.Xtest;
            if (doScale) { const s = scaleData(Xtr); Xtr = s.data; Xte = Xte.map(r => r.map((v, j) => (v - s.mu[j]) / s.sig[j])); }
            t0 = performance.now(); model.fit(Xtr, sp.ytrain); yp = model.predict(Xte); elapsed = performance.now() - t0;
            yt = sp.ytest;
          }

          const acc = accuracy(yt, yp), wf = weightedF1(yt, yp);
          results.push({ name, acc, f1: wf.f1, prec: wf.precision, rec: wf.recall, time: elapsed, yt, yp });
          log(`  ${name.padEnd(24)} Acc=${round4(acc)}  Prec=${round4(wf.precision)}  Rec=${round4(wf.recall)}  F1=${round4(wf.f1)}  ${(elapsed / 1e3).toFixed(2)}s`);
        } catch (e) {
          log(`<span class="r-err">  ${name}: ERROR – ${e.message}</span>`);
        }
      }

      if (results.length) {
        const best = results.reduce((a, b) => a.f1 > b.f1 ? a : b);
        log(`<span class="r-sep">${'-'.repeat(72)}</span>`);
        log(`<span class="r-best">Best: ${best.name}  F1=${round4(best.f1)}</span>`);

        // Confusion matrix chart
        const cm = confusionMatrix(best.yt, best.yp);
        const heatData = [{
          z: cm.matrix, x: classLabels, y: classLabels,
          type: 'heatmap', colorscale: [[0, '#1e293b'], [1, '#a855f7']],
          text: cm.matrix.map(r => r.map(v => String(v))), texttemplate: '%{text}', showscale: false,
        }];
        const layout = darkLayout({ title: `Confusion Matrix – ${best.name} (F1=${round4(best.f1)})`, xaxis: { title: 'Predicted', gridcolor: '#334155' }, yaxis: { title: 'Actual', autorange: 'reversed', gridcolor: '#334155' } });
        Plotly.newPlot('clf-chart', heatData, layout, { responsive: true });
      }
      this.setStatus('Classification complete.');
    }, 50);
  }

  /* ── Regression ──────────────────────────────────────────── */
  runRegression() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const target = document.getElementById('reg-target').value;
    const selected = [...document.querySelectorAll('#reg-algo-list input:checked')].map(el => el.dataset.algo);
    if (!selected.length) { alert('Select at least one algorithm.'); return; }
    const evalMode = document.querySelector('input[name="reg-eval"]:checked').value;
    const testPct = +document.getElementById('reg-test').value / 100;
    const folds = +document.getElementById('reg-folds').value;
    const doScale = document.getElementById('reg-scale').checked;

    let prep;
    try { prep = this._prepareData(target, 'regression'); } catch (e) { alert(e.message); return; }
    let { X, y, featCols } = prep;

    if (X.length > 5000) {
      const idx = seededShuffle([...Array(X.length).keys()], 42).slice(0, 5000);
      X = idx.map(i => X[i]); y = idx.map(i => y[i]);
    }

    const box = document.getElementById('reg-results');
    box.innerHTML = '';
    const log = t => box.innerHTML += t + '\n';
    log(`<span class="r-head">Regression  |  Target: ${target}  |  Mode: ${evalMode}</span>`);
    log(`<span class="r-head">Instances: ${X.length}  |  Features: ${featCols.length}</span>`);
    log(`<span class="r-sep">${'='.repeat(72)}</span>`);
    this.setStatus('Running regressors...');

    const results = [];
    setTimeout(() => {
      for (const name of selected) {
        const factory = this.regressorDefs[name]; if (!factory) continue;
        try {
          const model = factory();
          let yt, yp, t0, elapsed;

          if (evalMode === 'cv') {
            const cvFolds = kFoldSplit(X.length, folds, 42);
            const allPred = Array(X.length);
            t0 = performance.now();
            for (const { train, test } of cvFolds) {
              let Xtr = train.map(i => X[i]), ytr = train.map(i => y[i]);
              let Xte = test.map(i => X[i]);
              if (doScale) { const s = scaleData(Xtr); Xtr = s.data; Xte = Xte.map(r => r.map((v, j) => (v - s.mu[j]) / s.sig[j])); }
              model.fit(Xtr, ytr);
              const pr = model.predict(Xte);
              test.forEach((idx, i) => allPred[idx] = pr[i]);
            }
            elapsed = performance.now() - t0;
            yt = y; yp = allPred;
          } else if (evalMode === 'train') {
            let Xs = X;
            if (doScale) Xs = scaleData(X).data;
            t0 = performance.now(); model.fit(Xs, y); yp = model.predict(Xs); elapsed = performance.now() - t0;
            yt = y;
          } else {
            const sp = trainTestSplit(X, y, testPct, 42);
            let Xtr = sp.Xtrain, Xte = sp.Xtest;
            if (doScale) { const s = scaleData(Xtr); Xtr = s.data; Xte = Xte.map(r => r.map((v, j) => (v - s.mu[j]) / s.sig[j])); }
            t0 = performance.now(); model.fit(Xtr, sp.ytrain); yp = model.predict(Xte); elapsed = performance.now() - t0;
            yt = sp.ytest;
          }

          const r2 = r2Score(yt, yp), rm = rmseScore(yt, yp), ma = maeScore(yt, yp);
          results.push({ name, r2, rmse: rm, mae: ma, time: elapsed, yt, yp });
          log(`  ${name.padEnd(24)} R2=${round4(r2)}  RMSE=${round4(rm)}  MAE=${round4(ma)}  ${(elapsed / 1e3).toFixed(2)}s`);
        } catch (e) {
          log(`<span class="r-err">  ${name}: ERROR – ${e.message}</span>`);
        }
      }

      if (results.length) {
        const best = results.reduce((a, b) => a.r2 > b.r2 ? a : b);
        log(`<span class="r-sep">${'-'.repeat(72)}</span>`);
        log(`<span class="r-best">Best: ${best.name}  R2=${round4(best.r2)}</span>`);

        // Actual vs Predicted scatter
        Plotly.newPlot('reg-chart', [{
          x: best.yt, y: best.yp, mode: 'markers', type: 'scatter',
          marker: { size: 4, color: '#a855f7', opacity: .5 }, name: 'Predictions',
        }, {
          x: [Math.min(...best.yt), Math.max(...best.yt)],
          y: [Math.min(...best.yt), Math.max(...best.yt)],
          mode: 'lines', line: { color: '#ef4444', dash: 'dash', width: 2 }, name: 'Ideal',
        }], darkLayout({
          title: `Actual vs Predicted – ${best.name} (R2=${round4(best.r2)})`,
          xaxis: { title: 'Actual', gridcolor: '#334155' },
          yaxis: { title: 'Predicted', gridcolor: '#334155' },
        }), { responsive: true });
      }
      this.setStatus('Regression complete.');
    }, 50);
  }

  /* ── Clustering ──────────────────────────────────────────── */
  runClustering() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const k = +document.getElementById('clu-k').value;
    const doScale = document.getElementById('clu-scale').checked;
    const featChecks = document.querySelectorAll('#clu-feat-list input:checked');
    const feats = [...featChecks].map(el => el.dataset.feat);
    if (!feats.length) { alert('Select numeric features.'); return; }

    let X = this.data.map(r => feats.map(c => r[c])).filter(r => r.every(v => typeof v === 'number' && !isNaN(v)));
    if (X.length < k) { alert('Not enough valid rows.'); return; }
    if (X.length > 5000) X = seededShuffle(X, 42).slice(0, 5000);
    if (doScale) X = scaleData(X).data;

    const box = document.getElementById('clu-results');
    box.innerHTML = '';
    const log = t => box.innerHTML += t + '\n';
    this.setStatus('Running K-Means...');

    setTimeout(() => {
      const t0 = performance.now();
      const km = new KMeansModel(k, 100);
      const labels = km.fit(X);
      const elapsed = performance.now() - t0;
      const sil = silhouetteScore(X, labels);

      const counts = {};
      labels.forEach(l => counts[l] = (counts[l] || 0) + 1);

      log(`<span class="r-head">K-Means  |  k=${k}  |  Features: ${feats.length}</span>`);
      log(`<span class="r-sep">${'='.repeat(50)}</span>`);
      log(`Silhouette: ${round4(sil)}`);
      log(`Inertia: ${round4(km.inertia)}`);
      log(`Time: ${(elapsed / 1e3).toFixed(2)}s`);
      log(`\nCluster sizes:`);
      for (const [c, n] of Object.entries(counts).sort((a, b) => a[0] - b[0])) log(`  Cluster ${c}: ${n} samples`);

      // PCA-like 2D scatter (just use first 2 features or 2 PCs)
      let x0, x1;
      if (X[0].length >= 2) { x0 = X.map(r => r[0]); x1 = X.map(r => r[1]); }
      else { x0 = X.map(r => r[0]); x1 = X.map(() => 0); }

      const traces = [];
      for (const c of Object.keys(counts).sort()) {
        const idx = labels.map((l, i) => l === +c ? i : -1).filter(i => i >= 0);
        traces.push({
          x: idx.map(i => x0[i]), y: idx.map(i => x1[i]),
          mode: 'markers', type: 'scatter', name: `Cluster ${c}`,
          marker: { size: 4, opacity: .6 },
        });
      }
      Plotly.newPlot('clu-chart', traces, darkLayout({
        title: `K-Means (k=${k}, Sil=${round4(sil)})`,
        xaxis: { title: feats[0] || 'Dim 1', gridcolor: '#334155' },
        yaxis: { title: feats[1] || 'Dim 2', gridcolor: '#334155' },
      }), { responsive: true });

      this.setStatus('Clustering complete.');
    }, 50);
  }

  runElbow() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const doScale = document.getElementById('clu-scale').checked;
    const feats = [...document.querySelectorAll('#clu-feat-list input:checked')].map(el => el.dataset.feat);
    if (!feats.length) { alert('Select features.'); return; }
    let X = this.data.map(r => feats.map(c => r[c])).filter(r => r.every(v => typeof v === 'number' && !isNaN(v)));
    if (X.length > 3000) X = seededShuffle(X, 42).slice(0, 3000);
    if (doScale) X = scaleData(X).data;

    this.setStatus('Computing elbow plot...');
    setTimeout(() => {
      const ks = [], inertias = [], sils = [];
      for (let k = 2; k <= 10; k++) {
        const km = new KMeansModel(k, 80);
        km.fit(X);
        ks.push(k); inertias.push(km.inertia);
        sils.push(silhouetteScore(X, km.labels));
      }
      Plotly.newPlot('clu-chart', [
        { x: ks, y: inertias, name: 'Inertia', yaxis: 'y1', mode: 'lines+markers', marker: { color: '#a855f7' } },
        { x: ks, y: sils, name: 'Silhouette', yaxis: 'y2', mode: 'lines+markers', marker: { color: '#22c55e' } },
      ], darkLayout({
        title: 'Elbow Method & Silhouette',
        xaxis: { title: 'k', gridcolor: '#334155' },
        yaxis: { title: 'Inertia', gridcolor: '#334155', side: 'left' },
        yaxis2: { title: 'Silhouette', gridcolor: '#334155', side: 'right', overlaying: 'y' },
        legend: { x: .5, y: 1.12, orientation: 'h' },
      }), { responsive: true });
      this.setStatus('Elbow plot complete.');
    }, 50);
  }

  /* ── Visualization ───────────────────────────────────────── */
  plotViz() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const type = document.getElementById('viz-type').value;
    const xCol = document.getElementById('viz-x').value;
    const yCol = document.getElementById('viz-y').value;
    const hueCol = document.getElementById('viz-hue').value || null;
    const chartEl = 'viz-chart';

    try {
      if (type === 'histogram') {
        const vals = this.data.map(r => r[xCol]).filter(v => v != null);
        if (hueCol) {
          const cats = unique(this.data.map(r => r[hueCol]).filter(v => v != null)).slice(0, 10);
          const traces = cats.map(c => ({
            x: this.data.filter(r => r[hueCol] === c).map(r => r[xCol]),
            type: 'histogram', name: String(c), opacity: .7,
          }));
          Plotly.newPlot(chartEl, traces, darkLayout({ title: `Histogram – ${xCol}`, barmode: 'overlay', xaxis: { title: xCol, gridcolor: '#334155' } }), { responsive: true });
        } else {
          Plotly.newPlot(chartEl, [{ x: vals, type: 'histogram', marker: { color: '#a855f7' } }],
            darkLayout({ title: `Histogram – ${xCol}`, xaxis: { title: xCol, gridcolor: '#334155' } }), { responsive: true });
        }
      } else if (type === 'scatter') {
        if (hueCol) {
          const cats = unique(this.data.map(r => r[hueCol]).filter(v => v != null)).slice(0, 10);
          const traces = cats.map(c => {
            const sub = this.data.filter(r => r[hueCol] === c);
            return { x: sub.map(r => r[xCol]), y: sub.map(r => r[yCol]), mode: 'markers', type: 'scatter', name: String(c), marker: { size: 5, opacity: .6 } };
          });
          Plotly.newPlot(chartEl, traces, darkLayout({ title: `Scatter – ${xCol} vs ${yCol}`, xaxis: { title: xCol, gridcolor: '#334155' }, yaxis: { title: yCol, gridcolor: '#334155' } }), { responsive: true });
        } else {
          Plotly.newPlot(chartEl, [{ x: this.data.map(r => r[xCol]), y: this.data.map(r => r[yCol]), mode: 'markers', type: 'scatter', marker: { size: 4, color: '#a855f7', opacity: .4 } }],
            darkLayout({ title: `Scatter – ${xCol} vs ${yCol}`, xaxis: { title: xCol, gridcolor: '#334155' }, yaxis: { title: yCol, gridcolor: '#334155' } }), { responsive: true });
        }
      } else if (type === 'box') {
        if (hueCol) {
          const cats = unique(this.data.map(r => r[hueCol]).filter(v => v != null)).slice(0, 15);
          const traces = cats.map(c => ({
            y: this.data.filter(r => r[hueCol] === c).map(r => r[xCol]),
            type: 'box', name: String(c),
          }));
          Plotly.newPlot(chartEl, traces, darkLayout({ title: `Box Plot – ${xCol} by ${hueCol}` }), { responsive: true });
        } else {
          Plotly.newPlot(chartEl, [{ y: this.data.map(r => r[xCol]), type: 'box', name: xCol, marker: { color: '#a855f7' } }],
            darkLayout({ title: `Box Plot – ${xCol}` }), { responsive: true });
        }
      } else if (type === 'heatmap') {
        const numCols = this.columns.filter(c => this.types[c] === 'numeric').slice(0, 15);
        const corr = [];
        for (const c1 of numCols) {
          const row = [];
          for (const c2 of numCols) {
            const v1 = this.data.map(r => r[c1]).filter((_, i) => typeof this.data[i][c1] === 'number' && typeof this.data[i][c2] === 'number');
            const v2 = this.data.map(r => r[c2]).filter((_, i) => typeof this.data[i][c1] === 'number' && typeof this.data[i][c2] === 'number');
            const n = Math.min(v1.length, v2.length);
            const m1 = mean(v1.slice(0, n)), m2 = mean(v2.slice(0, n));
            const s1 = std(v1.slice(0, n)) || 1, s2 = std(v2.slice(0, n)) || 1;
            let cov = 0; for (let i = 0; i < n; i++) cov += (v1[i] - m1) * (v2[i] - m2);
            row.push(round4(n > 1 ? cov / ((n - 1) * s1 * s2) : 0));
          }
          corr.push(row);
        }
        Plotly.newPlot(chartEl, [{
          z: corr, x: numCols, y: numCols, type: 'heatmap',
          colorscale: [[0, '#3b82f6'], [.5, '#1e293b'], [1, '#ef4444']],
          zmin: -1, zmax: 1,
          text: corr.map(r => r.map(v => String(v))), texttemplate: numCols.length <= 12 ? '%{text}' : '',
        }], darkLayout({ title: 'Correlation Heatmap' }), { responsive: true });
      } else if (type === 'bar') {
        const freq = {};
        this.data.forEach(r => { const v = String(r[xCol] ?? ''); freq[v] = (freq[v] || 0) + 1; });
        const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 25);
        Plotly.newPlot(chartEl, [{ x: sorted.map(s => s[0]), y: sorted.map(s => s[1]), type: 'bar', marker: { color: '#a855f7' } }],
          darkLayout({ title: `Bar Chart – ${xCol}`, xaxis: { title: xCol, gridcolor: '#334155' }, yaxis: { title: 'Count', gridcolor: '#334155' } }), { responsive: true });
      } else if (type === 'dist-grid') {
        const numCols = this.columns.filter(c => this.types[c] === 'numeric').slice(0, 12);
        const traces = numCols.map((c, i) => ({
          x: this.data.map(r => r[c]).filter(v => typeof v === 'number' && !isNaN(v)),
          type: 'histogram', name: c,
          xaxis: `x${i + 1}`, yaxis: `y${i + 1}`,
          marker: { color: DARK_LAYOUT.colorway[i % DARK_LAYOUT.colorway.length] },
        }));
        const n = numCols.length, cols = Math.min(n, 4), rows = Math.ceil(n / cols);
        const layout = darkLayout({ title: 'Distribution Grid', showlegend: false });
        for (let i = 0; i < n; i++) {
          const r = Math.floor(i / cols), c = i % cols;
          const ax = i === 0 ? '' : String(i + 1);
          layout[`xaxis${ax}`] = { domain: [c / cols + .02, (c + 1) / cols - .02], row: r, gridcolor: '#334155', title: numCols[i] };
          layout[`yaxis${ax}`] = { domain: [1 - (r + 1) / rows + .04, 1 - r / rows - .04], gridcolor: '#334155' };
        }
        layout.grid = { rows, columns: cols, pattern: 'independent' };
        Plotly.newPlot(chartEl, traces, layout, { responsive: true });
      }
    } catch (e) {
      alert('Chart error: ' + e.message);
    }
  }

  /* ── Feature Selection ───────────────────────────────────── */
  runSelection() {
    if (!this.data) { alert('Load a dataset first.'); return; }
    const target = document.getElementById('sel-target').value;
    const method = document.getElementById('sel-method').value;
    const k = +document.getElementById('sel-k').value;
    const task = document.querySelector('input[name="sel-task"]:checked').value;

    let prep;
    try { prep = this._prepareData(target, task); } catch (e) { alert(e.message); return; }
    const { X, y, featCols } = prep;

    const box = document.getElementById('sel-results');
    box.innerHTML = '';
    const log = t => box.innerHTML += t + '\n';
    log(`<span class="r-head">Feature Selection  |  Target: ${target}  |  Method: ${method}</span>`);
    log(`<span class="r-sep">${'='.repeat(60)}</span>`);
    this.setStatus('Running feature selection...');

    setTimeout(() => {
      let ranked = [];
      try {
        if (method === 'correlation') {
          for (const [j, col] of featCols.entries()) {
            const fv = X.map(r => r[j]), m1 = mean(fv), s1 = std(fv) || 1, m2 = mean(y), s2 = std(y) || 1;
            let cov = 0; for (let i = 0; i < fv.length; i++) cov += (fv[i] - m1) * (y[i] - m2);
            ranked.push([col, Math.abs(cov / ((fv.length - 1) * s1 * s2))]);
          }
        } else if (method === 'variance') {
          for (const [j, col] of featCols.entries()) {
            ranked.push([col, variance(X.map(r => r[j]))]);
          }
        } else if (method === 'importance') {
          const rf = task === 'classification'
            ? new RandomForestClassifier(15, 8)
            : (() => { const dt = new DecisionTreeRegressor(10, 5); return dt; })();
          // For RF importance, train a decision tree and get importances
          const Xs = scaleData(X).data;
          const dt = new DecisionTreeClassifier(12, 5);
          dt.fit(Xs, y);
          const imp = dt.getImportances();
          for (const [j, col] of featCols.entries()) ranked.push([col, imp[j] || 0]);
        }
      } catch (e) {
        log(`<span class="r-err">ERROR: ${e.message}</span>`);
        return;
      }

      ranked.sort((a, b) => b[1] - a[1]);
      const top = ranked.slice(0, k);

      log(`\n${'Rank'.padEnd(6)}${'Feature'.padEnd(30)}Score`);
      log(`<span class="r-sep">${'-'.repeat(50)}</span>`);
      top.forEach(([name, val], i) => {
        const cls = i === 0 ? 'r-best' : '';
        log(`<span class="${cls}">  ${String(i + 1).padEnd(4)} ${name.padEnd(30)} ${round4(val)}</span>`);
      });

      // Bar chart
      Plotly.newPlot('sel-chart', [{
        y: top.map(t => t[0]).reverse(),
        x: top.map(t => t[1]).reverse(),
        type: 'bar', orientation: 'h',
        marker: { color: '#a855f7' },
      }], darkLayout({
        title: `Feature Selection – ${method}`,
        xaxis: { title: 'Score', gridcolor: '#334155' },
        yaxis: { gridcolor: '#334155' },
        margin: { l: 150 },
      }), { responsive: true });

      this.setStatus('Feature selection complete.');
    }, 50);
  }

  /* ── Utility ─────────────────────────────────────────────── */
  toggleAlgos(prefix, state) {
    document.querySelectorAll(`#${prefix}-algo-list input`).forEach(el => el.checked = state);
  }

  clearResults(prefix) {
    const box = document.getElementById(prefix + '-results');
    if (box) box.innerHTML = '';
    const chart = document.getElementById(prefix + '-chart');
    if (chart) Plotly.purge(chart);
  }

  setStatus(msg) { document.getElementById('status-text').textContent = msg; }
}

/* ────────────────────────────────────────────────────────────────
   9. INITIALIZATION
   ──────────────────────────────────────────────────────────────── */
const app = new PyWekaApp();
