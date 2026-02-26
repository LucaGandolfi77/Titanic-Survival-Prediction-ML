/* ═══════════════════════════════════════════════════════════════════════════
   operators_eval.js – Evaluation operators: Apply Model, Performance
   (Classification/Regression/Clustering), Cross Validation, Feature
   Importance.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Apply Model ──────────────────────────────────────────────────────────── */
registerOperator("Apply Model", () => {
  const op = new Operator("Apply Model", OpCategory.EVALUATION,
    [new Port("model", PortType.MODEL), new Port("data", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET), new Port("model_out", PortType.MODEL)],
    [], "Apply a trained model to new data to produce predictions."
  );
  op.execute = function(inp) {
    const model = inp.model;
    const df = inp.data;
    if (!model || !df) throw new Error("Apply Model: missing model or data input.");

    const feats = model._featureNames || [];
    const X = df.toMatrix(feats);

    if (model._type === "clustering") {
      const labels = model.predict(X);
      return { out: df.addColumn("cluster", labels.map(String)), model_out: model };
    }

    const preds = model.predict(X);
    const predCol = "prediction(" + (model._labelCol || "label") + ")";
    return { out: df.addColumn(predCol, preds), model_out: model };
  };
  return op;
});

/* ── Performance Classification ───────────────────────────────────────────── */
registerOperator("Performance Classification", () => {
  const op = new Operator("Performance Classification", OpCategory.EVALUATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET), new Port("performance", PortType.PERFORMANCE)],
    [], "Evaluate classification predictions vs actual labels."
  );
  op.execute = function(inp) {
    const df = inp.in;
    // Find prediction column and label column
    const predCol = df.columns.find(c => c.startsWith("prediction("));
    const labelColName = predCol ? predCol.match(/prediction\((.+)\)/)?.[1] : null;
    const labelCol = labelColName || df.labelCol();
    if (!predCol || !labelCol) throw new Error("Performance Classification: missing prediction or label column.");

    const yTrue = df.toArray(labelCol);
    const yPred = df.toArray(predCol);

    const accuracy  = Metrics.accuracy(yTrue, yPred);
    const precision = Metrics.precision(yTrue, yPred);
    const recall    = Metrics.recall(yTrue, yPred);
    const f1        = Metrics.f1(yTrue, yPred);
    const cm        = Metrics.confusionMatrix(yTrue, yPred);

    const perf = {
      type: "classification",
      accuracy, precision, recall, f1,
      confusion_matrix: cm,
    };
    return { out: df, performance: perf };
  };
  return op;
});

/* ── Performance Regression ───────────────────────────────────────────────── */
registerOperator("Performance Regression", () => {
  const op = new Operator("Performance Regression", OpCategory.EVALUATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET), new Port("performance", PortType.PERFORMANCE)],
    [], "Evaluate regression predictions vs actual values."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const predCol = df.columns.find(c => c.startsWith("prediction("));
    const labelColName = predCol ? predCol.match(/prediction\((.+)\)/)?.[1] : null;
    const labelCol = labelColName || df.labelCol();
    if (!predCol || !labelCol) throw new Error("Performance Regression: missing prediction or label column.");

    const yTrue = df.toArray(labelCol).map(Number);
    const yPred = df.toArray(predCol).map(Number);

    const perf = {
      type: "regression",
      mse:  Metrics.mse(yTrue, yPred),
      rmse: Metrics.rmse(yTrue, yPred),
      mae:  Metrics.mae(yTrue, yPred),
      r2:   Metrics.r2(yTrue, yPred),
    };
    return { out: df, performance: perf };
  };
  return op;
});

/* ── Performance Clustering ───────────────────────────────────────────────── */
registerOperator("Performance Clustering", () => {
  const op = new Operator("Performance Clustering", OpCategory.EVALUATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET), new Port("performance", PortType.PERFORMANCE)],
    [], "Evaluate clustering quality using silhouette score."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const clusterCol = df.columns.find(c => c === "cluster") || df.columns[df.columns.length - 1];
    const numCols = df.numericCols().filter(c => c !== clusterCol);
    const X = df.toMatrix(numCols);
    const labels = df.toArray(clusterCol).map(v => Number(v) || 0);

    const nClusters = new Set(labels).size;
    const silhouette = Metrics.silhouetteScore(X, labels);

    const perf = {
      type: "clustering",
      n_clusters: nClusters,
      silhouette_score: silhouette,
    };
    return { out: df, performance: perf };
  };
  return op;
});

/* ── Cross Validation ─────────────────────────────────────────────────────── */
registerOperator("Cross Validation", () => {
  const op = new Operator("Cross Validation", OpCategory.EVALUATION,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET), new Port("performance", PortType.PERFORMANCE)],
    [
      new ParamSpec("folds", ParamKind.INT, 5, "Number of folds"),
      new ParamSpec("model", ParamKind.CHOICE, "Logistic Regression", "Model to evaluate",
        ["Logistic Regression","Decision Tree","Random Forest","KNN","Naive Bayes",
         "Linear Regression","Ridge","Lasso"]),
    ], "K-fold cross validation."
  );
  op.execute = function(inp) {
    const df = inp.in;
    const k = this.params.folds;
    const modelName = this.params.model;

    const labelCol = df.labelCol();
    const numCols = df.numericCols().filter(c => c !== labelCol);
    const X = df.toMatrix(numCols);
    const y = df.toArray(labelCol);
    const isRegression = y.every(v => typeof v === "number");

    // Shuffle indices
    const rng = _seedRng(42);
    const idxs = Array.from({ length: X.length }, (_, i) => i).sort(() => rng() - 0.5);
    const foldSize = Math.ceil(idxs.length / k);
    const scores = [];

    for (let f = 0; f < k; f++) {
      const testIdx = new Set(idxs.slice(f * foldSize, (f + 1) * foldSize));
      const trainX = [], trainY = [], testX = [], testY = [];
      for (let i = 0; i < X.length; i++) {
        if (testIdx.has(i)) { testX.push(X[i]); testY.push(y[i]); }
        else { trainX.push(X[i]); trainY.push(y[i]); }
      }
      // Create & train model
      const ModelClass = ML_MODELS[modelName];
      if (!ModelClass) throw new Error(`Cross Validation: unknown model '${modelName}'.`);
      const model = new ModelClass(isRegression ? { task: "regression" } : {});
      model.fit(trainX, trainY);
      const preds = model.predict(testX);

      if (isRegression) {
        scores.push(Metrics.r2(testY, preds));
      } else {
        scores.push(Metrics.accuracy(testY, preds));
      }
    }

    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const std = Math.sqrt(scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length);

    const perf = {
      type: isRegression ? "regression_cv" : "classification_cv",
      metric: isRegression ? "R²" : "Accuracy",
      mean, std, folds: scores,
    };
    return { out: df, performance: perf };
  };
  return op;
});

/* ── Feature Importance ───────────────────────────────────────────────────── */
registerOperator("Feature Importance", () => {
  const op = new Operator("Feature Importance", OpCategory.EVALUATION,
    [new Port("model", PortType.MODEL)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [], "Extract feature importances from a tree-based model."
  );
  op.execute = function(inp) {
    const model = inp.model;
    if (!model || !model._featureNames) throw new Error("Feature Importance: invalid model.");
    const feats = model._featureNames;
    let imp;
    if (typeof model.featureImportances === "function") {
      imp = model.featureImportances(feats.length);
    } else {
      imp = feats.map(() => 1 / feats.length);
    }
    const data = feats.map((f, i) => [f, +(imp[i] || 0).toFixed(4)]);
    data.sort((a, b) => b[1] - a[1]);
    return { out: new DataFrame(["feature", "importance"], data) };
  };
  return op;
});
