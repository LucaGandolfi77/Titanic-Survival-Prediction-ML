/* ═══════════════════════════════════════════════════════════════════════════
   results.js – Results panel: data tables, metric cards, charts.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

class ResultsPanel {
  constructor(containerEl) {
    this.container = containerEl;
  }

  clear() {
    this.container.innerHTML = "";
  }

  /** Display all results from a process run */
  showAll(results, process) {
    this.clear();
    for (const [opId, outputs] of Object.entries(results)) {
      const op = process.operators[opId];
      if (!op) continue;
      for (const [portName, data] of Object.entries(outputs)) {
        if (data == null) continue;
        this._addResultSection(op.name, portName, data);
      }
    }
    if (!this.container.children.length) {
      this.container.innerHTML = '<p style="color:#999;text-align:center;margin-top:40px;">No results yet. Run a process first.</p>';
    }
  }

  _addResultSection(opName, portName, data) {
    const section = document.createElement("div");
    section.className = "result-section";

    const header = document.createElement("h3");
    header.className = "result-header";
    header.textContent = `${opName} → ${portName}`;
    section.appendChild(header);

    if (data instanceof DataFrame) {
      this._renderDataFrame(section, data);
      // Check for attached chart
      if (data._vizCanvas) {
        const chartWrap = document.createElement("div");
        chartWrap.className = "chart-wrap";
        chartWrap.appendChild(data._vizCanvas);
        section.appendChild(chartWrap);
      }
    } else if (data && typeof data === "object" && data.type) {
      this._renderPerformance(section, data);
    }

    this.container.appendChild(section);
  }

  _renderDataFrame(section, df) {
    const info = document.createElement("div");
    info.className = "result-info";
    info.textContent = `${df.nRows} rows × ${df.nCols} columns`;
    if (Object.keys(df.roles).length) {
      info.textContent += ` | Roles: ${Object.entries(df.roles).map(([k,v]) => `${k}=${v}`).join(", ")}`;
    }
    section.appendChild(info);

    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    table.className = "result-table";

    // Header
    const thead = document.createElement("thead");
    const hRow = document.createElement("tr");
    for (const col of df.columns) {
      const th = document.createElement("th");
      th.textContent = col;
      if (df.roles[col]) th.title = `Role: ${df.roles[col]}`;
      hRow.appendChild(th);
    }
    thead.appendChild(hRow);
    table.appendChild(thead);

    // Body (limit to 100 rows)
    const tbody = document.createElement("tbody");
    const maxRows = Math.min(df.nRows, 100);
    for (let i = 0; i < maxRows; i++) {
      const tr = document.createElement("tr");
      for (let j = 0; j < df.nCols; j++) {
        const td = document.createElement("td");
        const val = df.data[i][j];
        td.textContent = val == null ? "—" : typeof val === "number" ? Number(val.toFixed(4)) : val;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    section.appendChild(wrap);

    if (df.nRows > 100) {
      const more = document.createElement("div");
      more.className = "result-info";
      more.textContent = `Showing first 100 of ${df.nRows} rows.`;
      section.appendChild(more);
    }
  }

  _renderPerformance(section, perf) {
    const cards = document.createElement("div");
    cards.className = "metric-cards";

    if (perf.type === "classification") {
      this._addCard(cards, "Accuracy",  (perf.accuracy * 100).toFixed(2) + "%");
      this._addCard(cards, "Precision", (perf.precision * 100).toFixed(2) + "%");
      this._addCard(cards, "Recall",    (perf.recall * 100).toFixed(2) + "%");
      this._addCard(cards, "F1",        (perf.f1 * 100).toFixed(2) + "%");

      // Confusion matrix
      if (perf.confusion_matrix) {
        const cm = perf.confusion_matrix;
        const cmDiv = document.createElement("div");
        cmDiv.className = "confusion-matrix";
        cmDiv.innerHTML = "<h4>Confusion Matrix</h4>";
        const tbl = document.createElement("table");
        tbl.className = "result-table cm-table";
        const hRow = document.createElement("tr");
        hRow.appendChild(document.createElement("th"));
        for (const c of cm.classes) {
          const th = document.createElement("th");
          th.textContent = `Pred: ${c}`;
          hRow.appendChild(th);
        }
        tbl.appendChild(hRow);
        for (let i = 0; i < cm.classes.length; i++) {
          const tr = document.createElement("tr");
          const th = document.createElement("th");
          th.textContent = `True: ${cm.classes[i]}`;
          tr.appendChild(th);
          for (let j = 0; j < cm.classes.length; j++) {
            const td = document.createElement("td");
            td.textContent = cm.matrix[i][j];
            if (i === j) td.style.fontWeight = "bold";
            tr.appendChild(td);
          }
          tbl.appendChild(tr);
        }
        cmDiv.appendChild(tbl);
        section.appendChild(cmDiv);
      }
    } else if (perf.type === "regression") {
      this._addCard(cards, "MSE",  perf.mse.toFixed(4));
      this._addCard(cards, "RMSE", perf.rmse.toFixed(4));
      this._addCard(cards, "MAE",  perf.mae.toFixed(4));
      this._addCard(cards, "R²",   perf.r2.toFixed(4));
    } else if (perf.type === "clustering") {
      this._addCard(cards, "Clusters",  perf.n_clusters);
      this._addCard(cards, "Silhouette", perf.silhouette_score.toFixed(4));
    } else if (perf.type === "classification_cv" || perf.type === "regression_cv") {
      this._addCard(cards, perf.metric, perf.mean.toFixed(4) + " ± " + perf.std.toFixed(4));
      this._addCard(cards, "Folds", perf.folds.map(v => v.toFixed(3)).join(", "));
    }

    section.appendChild(cards);
  }

  _addCard(container, title, value) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<div class="metric-title">${title}</div><div class="metric-value">${value}</div>`;
    container.appendChild(card);
  }
}
