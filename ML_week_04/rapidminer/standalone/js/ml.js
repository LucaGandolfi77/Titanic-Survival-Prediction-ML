/* ═══════════════════════════════════════════════════════════════════════════
   ml.js – Pure-JavaScript ML algorithms for RapidMiner Lite.
   No external dependencies. Implements gradient descent, tree building,
   distance-based methods, clustering, dimensionality reduction.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Linear Algebra helpers ───────────────────────────────────────────────── */

const LA = {
  dot(a, b) { return a.reduce((s, v, i) => s + v * (b[i] || 0), 0); },
  add(a, b) { return a.map((v, i) => v + (b[i] || 0)); },
  sub(a, b) { return a.map((v, i) => v - (b[i] || 0)); },
  scale(a, k) { return a.map(v => v * k); },
  norm(a) { return Math.sqrt(a.reduce((s, v) => s + v * v, 0)); },
  zeros(n) { return new Array(n).fill(0); },
  eye(n) { return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0)); },

  matMul(A, B) {
    const m = A.length, n = B[0].length, p = B.length;
    const C = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < p; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  },

  transpose(A) {
    const m = A.length, n = A[0].length;
    return Array.from({ length: n }, (_, j) => Array.from({ length: m }, (_, i) => A[i][j]));
  },

  colMean(A) {
    const m = A.length, n = A[0].length;
    const mu = new Array(n).fill(0);
    for (const row of A) row.forEach((v, j) => mu[j] += v);
    return mu.map(v => v / m);
  },

  covMatrix(A) {
    const mu = LA.colMean(A);
    const m = A.length, n = mu.length;
    const C = Array.from({ length: n }, () => new Array(n).fill(0));
    for (const row of A) {
      const d = row.map((v, j) => v - mu[j]);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) C[i][j] += d[i] * d[j];
    }
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) C[i][j] /= m;
    return C;
  },

  argmax(arr) { return arr.reduce((bi, v, i, a) => v > a[bi] ? i : bi, 0); },
  argmin(arr) { return arr.reduce((bi, v, i, a) => v < a[bi] ? i : bi, 0); },
};

/* ── Sigmoid / softmax ────────────────────────────────────────────────────── */

function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

/* ── Train / test split utility ───────────────────────────────────────────── */

function trainTestSplit(X, y, ratio = 0.7, seed = 42) {
  const rng = _seedRng(seed);
  const n = X.length;
  const idxs = Array.from({ length: n }, (_, i) => i).sort(() => rng() - 0.5);
  const cut = Math.round(n * ratio);
  return {
    XTrain: idxs.slice(0, cut).map(i => X[i]),
    yTrain: idxs.slice(0, cut).map(i => y[i]),
    XTest:  idxs.slice(cut).map(i => X[i]),
    yTest:  idxs.slice(cut).map(i => y[i]),
  };
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOGISTIC REGRESSION (binary & multiclass via one-vs-rest)
   ═══════════════════════════════════════════════════════════════════════════ */

class LogisticRegression {
  constructor({ lr = 0.01, maxIter = 200, C = 1.0 } = {}) {
    this.lr = lr; this.maxIter = maxIter; this.C = C;
    this._models = []; this._classes = [];
  }

  fit(X, y) {
    this._classes = [...new Set(y)].sort();
    if (this._classes.length === 2) {
      this._models = [this._fitBinary(X, y, this._classes[1])];
    } else {
      this._models = this._classes.map(c => this._fitBinary(X, y, c));
    }
    return this;
  }

  _fitBinary(X, y, posClass) {
    const n = X.length, d = X[0].length;
    const w = LA.zeros(d);
    let b = 0;
    const labels = y.map(v => v === posClass ? 1 : 0);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const dw = LA.zeros(d);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const z = LA.dot(w, X[i]) + b;
        const p = sigmoid(z);
        const err = p - labels[i];
        for (let j = 0; j < d; j++) dw[j] += err * X[i][j];
        db += err;
      }
      for (let j = 0; j < d; j++) w[j] -= this.lr * (dw[j] / n + w[j] / this.C);
      b -= this.lr * db / n;
    }
    return { w, b, posClass };
  }

  predict(X) {
    if (this._classes.length === 2) {
      const { w, b } = this._models[0];
      return X.map(x => sigmoid(LA.dot(w, x) + b) >= 0.5 ? this._classes[1] : this._classes[0]);
    }
    return X.map(x => {
      const scores = this._models.map(m => sigmoid(LA.dot(m.w, x) + m.b));
      return this._classes[LA.argmax(scores)];
    });
  }

  predictProba(X) {
    if (this._classes.length === 2) {
      const { w, b } = this._models[0];
      return X.map(x => {
        const p = sigmoid(LA.dot(w, x) + b);
        return { [this._classes[0]]: 1 - p, [this._classes[1]]: p };
      });
    }
    return X.map(x => {
      const scores = this._models.map(m => sigmoid(LA.dot(m.w, x) + m.b));
      const sum = scores.reduce((a, b) => a + b, 0) || 1;
      const probs = {};
      this._classes.forEach((c, i) => { probs[c] = scores[i] / sum; });
      return probs;
    });
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   DECISION TREE (ID3 with entropy, supports classification & regression)
   ═══════════════════════════════════════════════════════════════════════════ */

class DecisionTree {
  constructor({ maxDepth = 10, minSamples = 2, task = "classification" } = {}) {
    this.maxDepth = maxDepth; this.minSamples = minSamples; this.task = task;
    this.root = null; this._classes = [];
  }

  fit(X, y) {
    this._classes = [...new Set(y)].sort();
    this.root = this._buildTree(X, y, 0);
    return this;
  }

  _entropy(y) {
    const counts = {};
    for (const v of y) counts[v] = (counts[v] || 0) + 1;
    const n = y.length;
    let h = 0;
    for (const c of Object.values(counts)) {
      const p = c / n; if (p > 0) h -= p * Math.log2(p);
    }
    return h;
  }

  _mse(y) {
    const m = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((s, v) => s + (v - m) ** 2, 0) / y.length;
  }

  _impurity(y) {
    return this.task === "regression" ? this._mse(y) : this._entropy(y);
  }

  _bestSplit(X, y) {
    const n = X.length, d = X[0].length;
    let bestGain = -Infinity, bestFeat = 0, bestThresh = 0;
    const parentImp = this._impurity(y);

    for (let f = 0; f < d; f++) {
      const vals = [...new Set(X.map(x => x[f]))].sort((a, b) => a - b);
      for (let t = 0; t < vals.length - 1; t++) {
        const thresh = (vals[t] + vals[t + 1]) / 2;
        const lIdx = [], rIdx = [];
        for (let i = 0; i < n; i++) (X[i][f] <= thresh ? lIdx : rIdx).push(i);
        if (lIdx.length < 1 || rIdx.length < 1) continue;
        const lY = lIdx.map(i => y[i]), rY = rIdx.map(i => y[i]);
        const gain = parentImp - (lY.length / n) * this._impurity(lY) - (rY.length / n) * this._impurity(rY);
        if (gain > bestGain) { bestGain = gain; bestFeat = f; bestThresh = thresh; }
      }
    }
    return { feature: bestFeat, threshold: bestThresh, gain: bestGain };
  }

  _buildTree(X, y, depth) {
    if (depth >= this.maxDepth || y.length < this.minSamples || this._impurity(y) === 0) {
      return { leaf: true, value: this._leafValue(y) };
    }
    const { feature, threshold, gain } = this._bestSplit(X, y);
    if (gain <= 0) return { leaf: true, value: this._leafValue(y) };

    const lIdx = [], rIdx = [];
    for (let i = 0; i < X.length; i++) (X[i][feature] <= threshold ? lIdx : rIdx).push(i);

    return {
      leaf: false, feature, threshold,
      left: this._buildTree(lIdx.map(i => X[i]), lIdx.map(i => y[i]), depth + 1),
      right: this._buildTree(rIdx.map(i => X[i]), rIdx.map(i => y[i]), depth + 1),
    };
  }

  _leafValue(y) {
    if (this.task === "regression") return y.reduce((a, b) => a + b, 0) / y.length;
    const counts = {};
    for (const v of y) counts[v] = (counts[v] || 0) + 1;
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }

  _predictOne(x, node) {
    if (node.leaf) return node.value;
    return x[node.feature] <= node.threshold
      ? this._predictOne(x, node.left) : this._predictOne(x, node.right);
  }

  predict(X) { return X.map(x => this._predictOne(x, this.root)); }

  featureImportances(nFeatures) {
    const imp = new Array(nFeatures).fill(0);
    const _walk = (node, weight) => {
      if (node.leaf) return;
      imp[node.feature] += weight;
      _walk(node.left, weight / 2);
      _walk(node.right, weight / 2);
    };
    _walk(this.root, 1);
    const s = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / s);
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   RANDOM FOREST
   ═══════════════════════════════════════════════════════════════════════════ */

class RandomForest {
  constructor({ nTrees = 10, maxDepth = 8, minSamples = 2, maxFeatures = "sqrt", task = "classification", seed = 42 } = {}) {
    this.nTrees = nTrees; this.maxDepth = maxDepth; this.minSamples = minSamples;
    this.maxFeatures = maxFeatures; this.task = task; this.seed = seed;
    this.trees = []; this._classes = []; this._featureSubsets = [];
  }

  fit(X, y) {
    this._classes = [...new Set(y)].sort();
    const rng = _seedRng(this.seed);
    const n = X.length, d = X[0].length;
    const mf = this.maxFeatures === "sqrt" ? Math.max(1, Math.round(Math.sqrt(d)))
             : this.maxFeatures === "log2" ? Math.max(1, Math.round(Math.log2(d))) : d;

    this.trees = [];
    this._featureSubsets = [];
    for (let t = 0; t < this.nTrees; t++) {
      // Bootstrap sample
      const idxs = Array.from({ length: n }, () => Math.floor(rng() * n));
      // Feature subset
      const allFeats = Array.from({ length: d }, (_, i) => i);
      allFeats.sort(() => rng() - 0.5);
      const feats = allFeats.slice(0, mf);
      this._featureSubsets.push(feats);

      const Xb = idxs.map(i => feats.map(f => X[i][f]));
      const yb = idxs.map(i => y[i]);
      const tree = new DecisionTree({ maxDepth: this.maxDepth, minSamples: this.minSamples, task: this.task });
      tree.fit(Xb, yb);
      this.trees.push(tree);
    }
    return this;
  }

  predict(X) {
    const preds = this.trees.map((tree, t) => {
      const feats = this._featureSubsets[t];
      const Xsub = X.map(x => feats.map(f => x[f]));
      return tree.predict(Xsub);
    });
    return X.map((_, i) => {
      const votes = preds.map(p => p[i]);
      if (this.task === "regression") return votes.reduce((a, b) => a + b, 0) / votes.length;
      const counts = {};
      for (const v of votes) counts[v] = (counts[v] || 0) + 1;
      return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    });
  }

  featureImportances(nFeatures) {
    const imp = new Array(nFeatures).fill(0);
    for (let t = 0; t < this.trees.length; t++) {
      const feats = this._featureSubsets[t];
      const treeImp = this.trees[t].featureImportances(feats.length);
      for (let j = 0; j < feats.length; j++) imp[feats[j]] += treeImp[j];
    }
    const s = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / s);
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   K-NEAREST NEIGHBOURS
   ═══════════════════════════════════════════════════════════════════════════ */

class KNN {
  constructor({ k = 5, task = "classification" } = {}) {
    this.k = k; this.task = task;
    this._X = null; this._y = null; this._classes = [];
  }

  fit(X, y) {
    this._X = X.map(r => [...r]);
    this._y = [...y];
    this._classes = [...new Set(y)].sort();
    return this;
  }

  predict(X) {
    return X.map(x => {
      const dists = this._X.map((xi, i) => ({
        d: Math.sqrt(xi.reduce((s, v, j) => s + (v - x[j]) ** 2, 0)),
        y: this._y[i],
      }));
      dists.sort((a, b) => a.d - b.d);
      const neighbors = dists.slice(0, this.k);
      if (this.task === "regression") return neighbors.reduce((s, n) => s + n.y, 0) / neighbors.length;
      const counts = {};
      for (const n of neighbors) counts[n.y] = (counts[n.y] || 0) + 1;
      return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    });
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   NAIVE BAYES (Gaussian)
   ═══════════════════════════════════════════════════════════════════════════ */

class NaiveBayes {
  constructor() {
    this._classes = []; this._priors = {}; this._stats = {};
  }

  fit(X, y) {
    this._classes = [...new Set(y)].sort();
    const n = X.length, d = X[0].length;
    for (const c of this._classes) {
      const rows = X.filter((_, i) => y[i] === c);
      this._priors[c] = rows.length / n;
      this._stats[c] = [];
      for (let j = 0; j < d; j++) {
        const vals = rows.map(r => r[j]);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const vari = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length || 1e-9;
        this._stats[c].push({ mean, variance: vari });
      }
    }
    return this;
  }

  _gaussian(x, mean, variance) {
    return Math.exp(-((x - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);
  }

  predict(X) {
    return X.map(x => {
      let bestClass = this._classes[0], bestLogP = -Infinity;
      for (const c of this._classes) {
        let logP = Math.log(this._priors[c]);
        for (let j = 0; j < x.length; j++) {
          const { mean, variance } = this._stats[c][j];
          logP += Math.log(this._gaussian(x[j], mean, variance) + 1e-300);
        }
        if (logP > bestLogP) { bestLogP = logP; bestClass = c; }
      }
      return bestClass;
    });
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LINEAR REGRESSION (OLS via gradient descent)
   ═══════════════════════════════════════════════════════════════════════════ */

class LinearRegression {
  constructor({ lr = 0.01, maxIter = 500 } = {}) {
    this.lr = lr; this.maxIter = maxIter;
    this.weights = null; this.bias = 0;
  }

  fit(X, y) {
    const n = X.length, d = X[0].length;
    this.weights = LA.zeros(d);
    this.bias = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      const dw = LA.zeros(d);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const pred = LA.dot(this.weights, X[i]) + this.bias;
        const err = pred - y[i];
        for (let j = 0; j < d; j++) dw[j] += err * X[i][j];
        db += err;
      }
      for (let j = 0; j < d; j++) this.weights[j] -= this.lr * dw[j] / n;
      this.bias -= this.lr * db / n;
    }
    return this;
  }

  predict(X) {
    return X.map(x => LA.dot(this.weights, x) + this.bias);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   RIDGE REGRESSION (L2)
   ═══════════════════════════════════════════════════════════════════════════ */

class RidgeRegression {
  constructor({ lr = 0.01, maxIter = 500, alpha = 1.0 } = {}) {
    this.lr = lr; this.maxIter = maxIter; this.alpha = alpha;
    this.weights = null; this.bias = 0;
  }

  fit(X, y) {
    const n = X.length, d = X[0].length;
    this.weights = LA.zeros(d);
    this.bias = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      const dw = LA.zeros(d);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const pred = LA.dot(this.weights, X[i]) + this.bias;
        const err = pred - y[i];
        for (let j = 0; j < d; j++) dw[j] += err * X[i][j];
        db += err;
      }
      for (let j = 0; j < d; j++)
        this.weights[j] -= this.lr * (dw[j] / n + this.alpha * this.weights[j] / n);
      this.bias -= this.lr * db / n;
    }
    return this;
  }

  predict(X) { return X.map(x => LA.dot(this.weights, x) + this.bias); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LASSO REGRESSION (L1)
   ═══════════════════════════════════════════════════════════════════════════ */

class LassoRegression {
  constructor({ lr = 0.01, maxIter = 500, alpha = 1.0 } = {}) {
    this.lr = lr; this.maxIter = maxIter; this.alpha = alpha;
    this.weights = null; this.bias = 0;
  }

  fit(X, y) {
    const n = X.length, d = X[0].length;
    this.weights = LA.zeros(d);
    this.bias = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      const dw = LA.zeros(d);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const pred = LA.dot(this.weights, X[i]) + this.bias;
        const err = pred - y[i];
        for (let j = 0; j < d; j++) dw[j] += err * X[i][j];
        db += err;
      }
      for (let j = 0; j < d; j++) {
        const reg = this.alpha * Math.sign(this.weights[j]) / n;
        this.weights[j] -= this.lr * (dw[j] / n + reg);
      }
      this.bias -= this.lr * db / n;
    }
    return this;
  }

  predict(X) { return X.map(x => LA.dot(this.weights, x) + this.bias); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   K-MEANS CLUSTERING
   ═══════════════════════════════════════════════════════════════════════════ */

class KMeans {
  constructor({ k = 3, maxIter = 100, seed = 42 } = {}) {
    this.k = k; this.maxIter = maxIter; this.seed = seed;
    this.centroids = []; this.labels = [];
  }

  fit(X) {
    const n = X.length, d = X[0].length;
    const rng = _seedRng(this.seed);
    // K-Means++ init
    this.centroids = [X[Math.floor(rng() * n)].slice()];
    while (this.centroids.length < this.k) {
      const dists = X.map(x => Math.min(...this.centroids.map(c =>
        x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0))));
      const total = dists.reduce((a, b) => a + b, 0);
      let r = rng() * total, cumSum = 0;
      for (let i = 0; i < n; i++) {
        cumSum += dists[i];
        if (cumSum >= r) { this.centroids.push(X[i].slice()); break; }
      }
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Assign
      this.labels = X.map(x =>
        LA.argmin(this.centroids.map(c => x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0))));
      // Update
      const newCentroids = Array.from({ length: this.k }, () => LA.zeros(d));
      const counts = new Array(this.k).fill(0);
      for (let i = 0; i < n; i++) {
        const l = this.labels[i];
        counts[l]++;
        for (let j = 0; j < d; j++) newCentroids[l][j] += X[i][j];
      }
      let moved = false;
      for (let c = 0; c < this.k; c++) {
        if (counts[c] === 0) continue;
        for (let j = 0; j < d; j++) {
          const nv = newCentroids[c][j] / counts[c];
          if (Math.abs(nv - this.centroids[c][j]) > 1e-8) moved = true;
          this.centroids[c][j] = nv;
        }
      }
      if (!moved) break;
    }
    return this;
  }

  predict(X) {
    return X.map(x =>
      LA.argmin(this.centroids.map(c => x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0))));
  }

  get inertia() {
    return 0; // placeholder
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   DBSCAN
   ═══════════════════════════════════════════════════════════════════════════ */

class DBSCAN {
  constructor({ eps = 0.5, minSamples = 5 } = {}) {
    this.eps = eps; this.minSamples = minSamples;
    this.labels = [];
  }

  fit(X) {
    const n = X.length;
    this.labels = new Array(n).fill(-1);
    let cluster = 0;
    const visited = new Array(n).fill(false);

    const dist = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
    const regionQuery = (idx) => {
      const neighbors = [];
      for (let j = 0; j < n; j++) if (dist(X[idx], X[j]) <= this.eps) neighbors.push(j);
      return neighbors;
    };

    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      visited[i] = true;
      const neighbors = regionQuery(i);
      if (neighbors.length < this.minSamples) { this.labels[i] = -1; continue; }
      this.labels[i] = cluster;
      const queue = [...neighbors];
      while (queue.length) {
        const q = queue.shift();
        if (!visited[q]) {
          visited[q] = true;
          const nn = regionQuery(q);
          if (nn.length >= this.minSamples) queue.push(...nn);
        }
        if (this.labels[q] === -1) this.labels[q] = cluster;
      }
      cluster++;
    }
    return this;
  }

  predict(X) { return this.labels; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   AGGLOMERATIVE CLUSTERING
   ═══════════════════════════════════════════════════════════════════════════ */

class AgglomerativeClustering {
  constructor({ k = 3, linkage = "average" } = {}) {
    this.k = k; this.linkage = linkage; this.labels = [];
  }

  fit(X) {
    const n = X.length;
    const dist = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

    // Each point starts as its own cluster
    let clusters = X.map((x, i) => ({ id: i, points: [i] }));

    while (clusters.length > this.k) {
      let bestDist = Infinity, bi = 0, bj = 1;
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          let d;
          const pa = clusters[i].points, pb = clusters[j].points;
          if (this.linkage === "single") {
            d = Infinity;
            for (const a of pa) for (const b of pb) d = Math.min(d, dist(X[a], X[b]));
          } else if (this.linkage === "complete") {
            d = 0;
            for (const a of pa) for (const b of pb) d = Math.max(d, dist(X[a], X[b]));
          } else { // average
            let sum = 0, cnt = 0;
            for (const a of pa) for (const b of pb) { sum += dist(X[a], X[b]); cnt++; }
            d = sum / cnt;
          }
          if (d < bestDist) { bestDist = d; bi = i; bj = j; }
        }
      }
      clusters[bi].points.push(...clusters[bj].points);
      clusters.splice(bj, 1);
    }

    this.labels = new Array(n).fill(0);
    clusters.forEach((c, ci) => { for (const p of c.points) this.labels[p] = ci; });
    return this;
  }

  predict(X) { return this.labels; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PCA (power iteration for top-k eigenvectors)
   ═══════════════════════════════════════════════════════════════════════════ */

class PCA {
  constructor({ nComponents = 2 } = {}) {
    this.nComponents = nComponents;
    this.components = []; this.mean = []; this.explainedVariance = [];
  }

  fit(X) {
    const mu = LA.colMean(X);
    this.mean = mu;
    const Xc = X.map(r => r.map((v, j) => v - mu[j]));
    const C = LA.covMatrix(Xc);
    const d = C.length;

    // Power iteration for each component
    this.components = [];
    this.explainedVariance = [];
    const deflatedC = C.map(r => [...r]);
    const rng = _seedRng(42);

    for (let comp = 0; comp < Math.min(this.nComponents, d); comp++) {
      let v = Array.from({ length: d }, () => rng() - 0.5);
      let norm = LA.norm(v); v = v.map(x => x / norm);

      for (let iter = 0; iter < 200; iter++) {
        const vNew = deflatedC.map(row => LA.dot(row, v));
        norm = LA.norm(vNew);
        if (norm < 1e-10) break;
        v = vNew.map(x => x / norm);
      }
      this.components.push(v);
      this.explainedVariance.push(norm);

      // Deflate
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++)
          deflatedC[i][j] -= norm * v[i] * v[j];
    }
    return this;
  }

  transform(X) {
    return X.map(row => {
      const centered = row.map((v, j) => v - this.mean[j]);
      return this.components.map(comp => LA.dot(centered, comp));
    });
  }

  fitTransform(X) { this.fit(X); return this.transform(X); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SVM (linear, simplified using hinge loss + SGD)
   ═══════════════════════════════════════════════════════════════════════════ */

class LinearSVM {
  constructor({ lr = 0.001, maxIter = 300, C = 1.0 } = {}) {
    this.lr = lr; this.maxIter = maxIter; this.C = C;
    this.weights = null; this.bias = 0; this._classes = [];
  }

  fit(X, y) {
    this._classes = [...new Set(y)].sort();
    // Binary: map to +1/-1
    const labels = y.map(v => v === this._classes[1] ? 1 : -1);
    const n = X.length, d = X[0].length;
    this.weights = LA.zeros(d);
    this.bias = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        const margin = labels[i] * (LA.dot(this.weights, X[i]) + this.bias);
        if (margin < 1) {
          for (let j = 0; j < d; j++)
            this.weights[j] += this.lr * (labels[i] * X[i][j] - this.weights[j] / (this.C * n));
          this.bias += this.lr * labels[i];
        } else {
          for (let j = 0; j < d; j++)
            this.weights[j] -= this.lr * this.weights[j] / (this.C * n);
        }
      }
    }
    return this;
  }

  predict(X) {
    return X.map(x => LA.dot(this.weights, x) + this.bias >= 0 ? this._classes[1] : this._classes[0]);
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   GRADIENT BOOSTING (simplified)
   ═══════════════════════════════════════════════════════════════════════════ */

class GradientBoosting {
  constructor({ nEstimators = 50, maxDepth = 3, lr = 0.1, task = "classification", seed = 42 } = {}) {
    this.nEstimators = nEstimators; this.maxDepth = maxDepth; this.lr = lr;
    this.task = task; this.seed = seed;
    this.trees = []; this._classes = []; this._basePred = 0;
  }

  fit(X, y) {
    if (this.task === "classification") {
      this._classes = [...new Set(y)].sort();
      // Binary classification with log loss
      const labels = y.map(v => v === this._classes[1] ? 1 : 0);
      const mean = labels.reduce((a, b) => a + b, 0) / labels.length;
      this._basePred = Math.log(mean / (1 - mean + 1e-10));
      let F = labels.map(() => this._basePred);

      for (let t = 0; t < this.nEstimators; t++) {
        const residuals = labels.map((y, i) => y - sigmoid(F[i]));
        const tree = new DecisionTree({ maxDepth: this.maxDepth, minSamples: 2, task: "regression" });
        tree.fit(X, residuals);
        const preds = tree.predict(X);
        F = F.map((f, i) => f + this.lr * preds[i]);
        this.trees.push(tree);
      }
    } else {
      const mean = y.reduce((a, b) => a + b, 0) / y.length;
      this._basePred = mean;
      let F = y.map(() => mean);

      for (let t = 0; t < this.nEstimators; t++) {
        const residuals = y.map((yi, i) => yi - F[i]);
        const tree = new DecisionTree({ maxDepth: this.maxDepth, minSamples: 2, task: "regression" });
        tree.fit(X, residuals);
        const preds = tree.predict(X);
        F = F.map((f, i) => f + this.lr * preds[i]);
        this.trees.push(tree);
      }
    }
    return this;
  }

  predict(X) {
    const rawScores = X.map(x => {
      let f = this._basePred;
      for (const tree of this.trees) f += this.lr * tree.predict([x])[0];
      return f;
    });
    if (this.task === "classification") {
      return rawScores.map(f => sigmoid(f) >= 0.5 ? this._classes[1] : this._classes[0]);
    }
    return rawScores;
  }

  featureImportances(nFeatures) {
    const imp = new Array(nFeatures).fill(0);
    for (const tree of this.trees) {
      const ti = tree.featureImportances(nFeatures);
      for (let j = 0; j < nFeatures; j++) imp[j] += ti[j];
    }
    const s = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map(v => v / s);
  }

  get classes() { return this._classes; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   METRICS
   ═══════════════════════════════════════════════════════════════════════════ */

const Metrics = {
  accuracy(yTrue, yPred) {
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) if (String(yTrue[i]) === String(yPred[i])) correct++;
    return correct / yTrue.length;
  },

  precision(yTrue, yPred, posClass = null) {
    const classes = posClass ? [posClass] : [...new Set(yTrue)];
    let totalPrec = 0;
    for (const c of classes) {
      let tp = 0, fp = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if (String(yPred[i]) === String(c)) {
          if (String(yTrue[i]) === String(c)) tp++; else fp++;
        }
      }
      totalPrec += tp / (tp + fp || 1);
    }
    return totalPrec / classes.length;
  },

  recall(yTrue, yPred, posClass = null) {
    const classes = posClass ? [posClass] : [...new Set(yTrue)];
    let totalRec = 0;
    for (const c of classes) {
      let tp = 0, fn = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if (String(yTrue[i]) === String(c)) {
          if (String(yPred[i]) === String(c)) tp++; else fn++;
        }
      }
      totalRec += tp / (tp + fn || 1);
    }
    return totalRec / classes.length;
  },

  f1(yTrue, yPred) {
    const p = Metrics.precision(yTrue, yPred);
    const r = Metrics.recall(yTrue, yPred);
    return 2 * p * r / (p + r || 1);
  },

  confusionMatrix(yTrue, yPred) {
    const classes = [...new Set([...yTrue, ...yPred])].sort();
    const mat = classes.map(() => classes.map(() => 0));
    for (let i = 0; i < yTrue.length; i++) {
      const ti = classes.indexOf(String(yTrue[i]));
      const pi = classes.indexOf(String(yPred[i]));
      if (ti >= 0 && pi >= 0) mat[ti][pi]++;
    }
    return { matrix: mat, classes };
  },

  mse(yTrue, yPred) {
    return yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0) / yTrue.length;
  },

  rmse(yTrue, yPred) {
    return Math.sqrt(Metrics.mse(yTrue, yPred));
  },

  mae(yTrue, yPred) {
    return yTrue.reduce((s, v, i) => s + Math.abs(v - yPred[i]), 0) / yTrue.length;
  },

  r2(yTrue, yPred) {
    const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
    const ssTot = yTrue.reduce((s, v) => s + (v - mean) ** 2, 0);
    const ssRes = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
    return 1 - ssRes / (ssTot || 1);
  },

  silhouetteScore(X, labels) {
    const n = X.length;
    if (n < 2) return 0;
    const dist = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
    const clusters = [...new Set(labels)];
    if (clusters.length < 2) return 0;
    let totalS = 0;
    for (let i = 0; i < n; i++) {
      const ci = labels[i];
      const sameCluster = [];
      for (let j = 0; j < n; j++) if (j !== i && labels[j] === ci) sameCluster.push(j);
      const a = sameCluster.length > 0
        ? sameCluster.reduce((s, j) => s + dist(X[i], X[j]), 0) / sameCluster.length : 0;
      let b = Infinity;
      for (const ck of clusters) {
        if (ck === ci) continue;
        const others = [];
        for (let j = 0; j < n; j++) if (labels[j] === ck) others.push(j);
        if (others.length === 0) continue;
        const avgDist = others.reduce((s, j) => s + dist(X[i], X[j]), 0) / others.length;
        b = Math.min(b, avgDist);
      }
      totalS += (b - a) / Math.max(a, b, 1e-10);
    }
    return totalS / n;
  },
};

/* ── ML registry (name → class) ──────────────────────────────────────────── */

const ML_MODELS = {
  "Logistic Regression":     LogisticRegression,
  "Decision Tree":           DecisionTree,
  "Random Forest":           RandomForest,
  "Gradient Boosting":       GradientBoosting,
  "SVM":                     LinearSVM,
  "KNN":                     KNN,
  "Naive Bayes":             NaiveBayes,
  "Linear Regression":       LinearRegression,
  "Ridge":                   RidgeRegression,
  "Lasso":                   LassoRegression,
  "KMeans":                  KMeans,
  "DBSCAN":                  DBSCAN,
  "Agglomerative":           AgglomerativeClustering,
  "PCA":                     PCA,
};
