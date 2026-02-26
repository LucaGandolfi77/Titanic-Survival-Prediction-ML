/* ═══════════════════════════════════════════════════════════════════════════
   operators_model.js – 13 model-training operators.
   Classification, Regression, Clustering – all backed by js/ml.js.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

function _prepClassification(df) {
  const labelCol = df.labelCol();
  const numCols = df.numericCols().filter(c => c !== labelCol);
  const X = df.toMatrix(numCols);
  const y = df.toArray(labelCol);
  return { X, y, featureNames: numCols, labelCol };
}

function _prepRegression(df) {
  const labelCol = df.labelCol();
  const numCols = df.numericCols().filter(c => c !== labelCol);
  const X = df.toMatrix(numCols);
  const y = df.toArray(labelCol).map(v => Number(v) || 0);
  return { X, y, featureNames: numCols, labelCol };
}

function _prepClustering(df) {
  const numCols = df.numericCols();
  const X = df.toMatrix(numCols);
  return { X, featureNames: numCols };
}

/* ── Logistic Regression ──────────────────────────────────────────────────── */
registerOperator("Logistic Regression", () => {
  const op = new Operator("Logistic Regression", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("C", ParamKind.FLOAT, 1.0, "Regularisation strength"),
      new ParamSpec("max_iter", ParamKind.INT, 200, "Max iterations"),
    ], "Train a logistic regression classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new LogisticRegression({ C: this.params.C, maxIter: this.params.max_iter });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Decision Tree ────────────────────────────────────────────────────────── */
registerOperator("Decision Tree", () => {
  const op = new Operator("Decision Tree", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("max_depth", ParamKind.INT, 10, "Max tree depth"),
      new ParamSpec("min_samples", ParamKind.INT, 2, "Min samples to split"),
    ], "Train a decision tree classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new DecisionTree({ maxDepth: this.params.max_depth, minSamples: this.params.min_samples, task: "classification" });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Random Forest ────────────────────────────────────────────────────────── */
registerOperator("Random Forest", () => {
  const op = new Operator("Random Forest", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("n_trees", ParamKind.INT, 10, "Number of trees"),
      new ParamSpec("max_depth", ParamKind.INT, 8, "Max tree depth"),
    ], "Train a random forest classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new RandomForest({ nTrees: this.params.n_trees, maxDepth: this.params.max_depth, task: "classification" });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Gradient Boosting ────────────────────────────────────────────────────── */
registerOperator("Gradient Boosting", () => {
  const op = new Operator("Gradient Boosting", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("n_estimators", ParamKind.INT, 50, "Number of boosting rounds"),
      new ParamSpec("max_depth", ParamKind.INT, 3, "Max tree depth"),
      new ParamSpec("learning_rate", ParamKind.FLOAT, 0.1, "Learning rate"),
    ], "Train a gradient boosting classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new GradientBoosting({
      nEstimators: this.params.n_estimators, maxDepth: this.params.max_depth,
      lr: this.params.learning_rate, task: "classification"
    });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── SVM ──────────────────────────────────────────────────────────────────── */
registerOperator("SVM", () => {
  const op = new Operator("SVM", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("C", ParamKind.FLOAT, 1.0, "Regularisation")],
    "Train a linear SVM classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new LinearSVM({ C: this.params.C });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── KNN ──────────────────────────────────────────────────────────────────── */
registerOperator("KNN", () => {
  const op = new Operator("KNN", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("k", ParamKind.INT, 5, "Number of neighbours")],
    "Train a K-Nearest Neighbours classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new KNN({ k: this.params.k, task: "classification" });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Naive Bayes ──────────────────────────────────────────────────────────── */
registerOperator("Naive Bayes", () => {
  const op = new Operator("Naive Bayes", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [], "Train a Gaussian Naive Bayes classifier."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepClassification(inp.training);
    const model = new NaiveBayes();
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "classification";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Linear Regression ────────────────────────────────────────────────────── */
registerOperator("Linear Regression", () => {
  const op = new Operator("Linear Regression", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("max_iter", ParamKind.INT, 500, "Max iterations")],
    "Train a linear regression model."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepRegression(inp.training);
    const model = new LinearRegression({ maxIter: this.params.max_iter });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "regression";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Ridge ────────────────────────────────────────────────────────────────── */
registerOperator("Ridge", () => {
  const op = new Operator("Ridge", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("alpha", ParamKind.FLOAT, 1.0, "Regularisation strength")],
    "Train a Ridge regression model."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepRegression(inp.training);
    const model = new RidgeRegression({ alpha: this.params.alpha });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "regression";
    return { model, out: inp.training };
  };
  return op;
});

/* ── Lasso ────────────────────────────────────────────────────────────────── */
registerOperator("Lasso", () => {
  const op = new Operator("Lasso", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("alpha", ParamKind.FLOAT, 1.0, "Regularisation strength")],
    "Train a Lasso regression model."
  );
  op.execute = function(inp) {
    const { X, y, featureNames, labelCol } = _prepRegression(inp.training);
    const model = new LassoRegression({ alpha: this.params.alpha });
    model.fit(X, y);
    model._featureNames = featureNames;
    model._labelCol = labelCol;
    model._type = "regression";
    return { model, out: inp.training };
  };
  return op;
});

/* ── KMeans ───────────────────────────────────────────────────────────────── */
registerOperator("KMeans", () => {
  const op = new Operator("KMeans", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("k", ParamKind.INT, 3, "Number of clusters")],
    "Train a K-Means clustering model."
  );
  op.execute = function(inp) {
    const { X, featureNames } = _prepClustering(inp.training);
    const model = new KMeans({ k: this.params.k });
    model.fit(X);
    model._featureNames = featureNames;
    model._type = "clustering";
    // Add cluster label to output
    const df = inp.training.addColumn("cluster", model.labels.map(String));
    return { model, out: df };
  };
  return op;
});

/* ── DBSCAN ───────────────────────────────────────────────────────────────── */
registerOperator("DBSCAN", () => {
  const op = new Operator("DBSCAN", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("eps", ParamKind.FLOAT, 0.5, "Epsilon neighbourhood radius"),
      new ParamSpec("min_samples", ParamKind.INT, 5, "Min samples in neighbourhood"),
    ], "Train a DBSCAN clustering model."
  );
  op.execute = function(inp) {
    const { X, featureNames } = _prepClustering(inp.training);
    const model = new DBSCAN({ eps: this.params.eps, minSamples: this.params.min_samples });
    model.fit(X);
    model._featureNames = featureNames;
    model._type = "clustering";
    const df = inp.training.addColumn("cluster", model.labels.map(String));
    return { model, out: df };
  };
  return op;
});

/* ── Agglomerative ────────────────────────────────────────────────────────── */
registerOperator("Agglomerative", () => {
  const op = new Operator("Agglomerative", OpCategory.MODEL,
    [new Port("training", PortType.EXAMPLE_SET)],
    [new Port("model", PortType.MODEL), new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("k", ParamKind.INT, 3, "Number of clusters"),
      new ParamSpec("linkage", ParamKind.CHOICE, "average", "Linkage", ["single","complete","average"]),
    ], "Train an agglomerative clustering model."
  );
  op.execute = function(inp) {
    const { X, featureNames } = _prepClustering(inp.training);
    const model = new AgglomerativeClustering({ k: this.params.k, linkage: this.params.linkage });
    model.fit(X);
    model._featureNames = featureNames;
    model._type = "clustering";
    const df = inp.training.addColumn("cluster", model.labels.map(String));
    return { model, out: df };
  };
  return op;
});
