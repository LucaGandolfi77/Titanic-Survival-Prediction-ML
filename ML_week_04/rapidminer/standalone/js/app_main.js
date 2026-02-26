/* ═══════════════════════════════════════════════════════════════════════════
   app_main.js – Main application controller.
   Wires together Canvas, Palette, Properties, Results panels.
   Manages toolbar actions, file I/O, undo/redo, AutoModel wizard.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

class App {
  constructor() {
    // State
    this.process = new Process("Untitled");
    this._undoStack = [];
    this._redoStack = [];
    this._runner = null;
    this._lastResults = null;

    // DOM references
    this._canvasEl   = document.getElementById("design-canvas");
    this._paletteEl  = document.getElementById("palette");
    this._searchEl   = document.getElementById("palette-search");
    this._propTitle  = document.getElementById("prop-title");
    this._propsEl    = document.getElementById("props");
    this._resultsEl  = document.getElementById("results");
    this._logEl      = document.getElementById("log");
    this._modalEl    = document.getElementById("modal-overlay");
    this._modalBody  = document.getElementById("modal-body");

    // Panels
    this.canvasRenderer = new CanvasRenderer(this._canvasEl, this);
    this.palette        = new PalettePanel(this._paletteEl, this._searchEl, this);
    this.properties     = new PropertiesPanel(this._propTitle, this._propsEl, this);
    this.results        = new ResultsPanel(this._resultsEl);

    this._bindToolbar();
    this._bindTabs();
    this._bindKeyboard();

    this.properties.clear();
    this.log("RapidMiner Lite ready. Drag operators from the palette to the canvas.");
  }

  /* ── Logging ────────────────────────────────────────────────────────── */

  log(msg) {
    const ts = new Date().toLocaleTimeString();
    this._logEl.textContent += `[${ts}] ${msg}\n`;
    this._logEl.scrollTop = this._logEl.scrollHeight;
  }

  /* ── Tabs ───────────────────────────────────────────────────────────── */

  _bindTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
        btn.classList.add("active");
        const panel = document.getElementById(btn.dataset.tab);
        if (panel) panel.classList.add("active");
        if (btn.dataset.tab === "tab-design") {
          requestAnimationFrame(() => this.canvasRenderer._resize());
        }
      });
    });
  }

  /* ── Toolbar ────────────────────────────────────────────────────────── */

  _bindToolbar() {
    document.getElementById("btn-new")?.addEventListener("click", () => this.newProcess());
    document.getElementById("btn-open")?.addEventListener("click", () => document.getElementById("file-open").click());
    document.getElementById("btn-save")?.addEventListener("click", () => this.saveProcess());
    document.getElementById("btn-run")?.addEventListener("click", () => this.runProcess());
    document.getElementById("btn-stop")?.addEventListener("click", () => this.stopProcess());
    document.getElementById("btn-undo")?.addEventListener("click", () => this.undo());
    document.getElementById("btn-redo")?.addEventListener("click", () => this.redo());
    document.getElementById("btn-clear")?.addEventListener("click", () => this.clearCanvas());
    document.getElementById("btn-automodel")?.addEventListener("click", () => this.showAutoModel());

    // Sample processes
    document.getElementById("sel-sample-process")?.addEventListener("change", e => {
      const name = e.target.value;
      if (name && SAMPLE_PROCESSES[name]) {
        this.process = Process.fromJSON(SAMPLE_PROCESSES[name]);
        this.canvasRenderer.selectedNode = null;
        this.properties.clear();
        this.canvasRenderer.fitView();
        this.pushUndo();
        this.log(`Loaded sample process: ${name}`);
      }
      e.target.value = "";
    });

    // Sample data
    document.getElementById("sel-sample-data")?.addEventListener("change", e => {
      const name = e.target.value;
      if (name && SAMPLE_DATASETS[name]) {
        const df = SAMPLE_DATASETS[name]();
        this.results.clear();
        this.results._addResultSection("Sample Data", name, df);
        this._switchTab("tab-results");
        this.log(`Loaded sample dataset: ${name} (${df.nRows} rows)`);
      }
      e.target.value = "";
    });

    // File inputs
    document.getElementById("file-open")?.addEventListener("change", e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        try {
          const json = JSON.parse(ev.target.result);
          this.process = Process.fromJSON(json);
          this.canvasRenderer.selectedNode = null;
          this.properties.clear();
          this.canvasRenderer.fitView();
          this.pushUndo();
          this.log(`Opened process: ${file.name}`);
        } catch (err) {
          this.log(`Error opening process: ${err.message}`);
        }
      };
      reader.readAsText(file);
      e.target.value = "";
    });

    document.getElementById("file-csv")?.addEventListener("change", e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        const csvText = ev.target.result;
        // If there's a ReadCSV operator selected, inject the text
        const selId = this.canvasRenderer.selectedNode;
        if (selId && this.process.operators[selId]?.name === "Read CSV") {
          this.process.operators[selId].params._csv_text = csvText;
          this.log(`CSV loaded into Read CSV operator (${file.name})`);
        } else {
          // Auto-create a Read CSV operator
          const id = this.process.addOperator("Read CSV", 100, 100);
          this.process.operators[id].params._csv_text = csvText;
          this.canvasRenderer.draw();
          this.pushUndo();
          this.log(`CSV loaded as new Read CSV operator (${file.name})`);
        }
      };
      reader.readAsText(file);
      e.target.value = "";
    });
  }

  /* ── Keyboard shortcuts ─────────────────────────────────────────────── */

  _bindKeyboard() {
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z") { e.preventDefault(); this.undo(); }
        if (e.key === "y") { e.preventDefault(); this.redo(); }
        if (e.key === "s") { e.preventDefault(); this.saveProcess(); }
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        const sel = this.canvasRenderer.selectedNode;
        if (sel) {
          this.process.removeOperator(sel);
          this.canvasRenderer.selectedNode = null;
          this.properties.clear();
          this.canvasRenderer.draw();
          this.pushUndo();
        }
      }
      if (e.key === "F5") { e.preventDefault(); this.runProcess(); }
      if (e.key === "Escape") { this.closeModal(); }
    });
  }

  /* ── Tab switching ──────────────────────────────────────────────────── */

  _switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-content").forEach(tc => {
      tc.classList.toggle("active", tc.id === tabId);
    });
  }

  /* ── Process operations ─────────────────────────────────────────────── */

  newProcess() {
    this.process = new Process("Untitled");
    this.canvasRenderer.selectedNode = null;
    this.properties.clear();
    this.results.clear();
    this._logEl.textContent = "";
    this._undoStack = [];
    this._redoStack = [];
    this.canvasRenderer.draw();
    this.log("New process created.");
  }

  saveProcess() {
    const json = JSON.stringify(this.process.toJSON(), null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${this.process.name || "process"}.rmp.json`; a.click();
    URL.revokeObjectURL(url);
    this.log("Process saved.");
  }

  clearCanvas() {
    this.process.operators = {};
    this.process.connections = [];
    this.process.positions = {};
    this.canvasRenderer.selectedNode = null;
    this.properties.clear();
    this.canvasRenderer.draw();
    this.pushUndo();
    this.log("Canvas cleared.");
  }

  addOperatorToProcess(opName, x, y) {
    const id = this.process.addOperator(opName, x, y);
    this.canvasRenderer.selectedNode = id;
    this.canvasRenderer.draw();
    this.onNodeSelected(id);
    this.pushUndo();
    this.log(`Added operator: ${opName}`);
    return id;
  }

  runProcess() {
    const opCount = Object.keys(this.process.operators).length;
    if (!opCount) { this.log("No operators to run."); return; }

    this._logEl.textContent = "";
    this._runner = new ProcessRunner(this.process, {
      onLog: msg => this.log(msg),
      onProgress: (id, name, step, total) => {
        document.title = `[${step}/${total}] ${name} – RapidMiner Lite`;
      },
      onResult: () => {},
    });

    try {
      this._lastResults = this._runner.run();
      this.results.showAll(this._lastResults, this.process);
      this._switchTab("tab-results");
      document.title = "RapidMiner Lite";
    } catch (err) {
      this.log(`Process error: ${err.message}`);
      document.title = "RapidMiner Lite";
    }
  }

  stopProcess() {
    if (this._runner) {
      this._runner.stop();
      this.log("Stop requested.");
    }
  }

  /* ── Undo / Redo ────────────────────────────────────────────────────── */

  pushUndo() {
    this._undoStack.push(JSON.stringify(this.process.toJSON()));
    if (this._undoStack.length > 50) this._undoStack.shift();
    this._redoStack = [];
  }

  undo() {
    if (this._undoStack.length < 1) return;
    this._redoStack.push(JSON.stringify(this.process.toJSON()));
    const json = JSON.parse(this._undoStack.pop());
    this.process = Process.fromJSON(json);
    this.canvasRenderer.selectedNode = null;
    this.properties.clear();
    this.canvasRenderer.draw();
  }

  redo() {
    if (this._redoStack.length < 1) return;
    this._undoStack.push(JSON.stringify(this.process.toJSON()));
    const json = JSON.parse(this._redoStack.pop());
    this.process = Process.fromJSON(json);
    this.canvasRenderer.selectedNode = null;
    this.properties.clear();
    this.canvasRenderer.draw();
  }

  /* ── Node interaction callbacks ─────────────────────────────────────── */

  onNodeSelected(id) {
    if (id) this.properties.show(id);
    else this.properties.clear();
  }

  onNodeDblClick(id) {
    this.properties.show(id);
  }

  /* ── Modal ──────────────────────────────────────────────────────────── */

  showModal(content) {
    this._modalBody.innerHTML = "";
    if (typeof content === "string") this._modalBody.innerHTML = content;
    else this._modalBody.appendChild(content);
    this._modalEl.classList.add("active");
  }

  closeModal() {
    this._modalEl.classList.remove("active");
    this._modalBody.innerHTML = "";
  }

  /* ── AutoModel Wizard ───────────────────────────────────────────────── */

  showAutoModel() {
    const div = document.createElement("div");
    div.className = "automodel-wizard";
    div.innerHTML = `
      <h2>AutoModel Wizard</h2>
      <p>Automatically build a classification or regression pipeline.</p>
      <div class="prop-group">
        <label>Dataset</label>
        <select id="am-dataset">
          <option value="iris">Iris</option>
          <option value="titanic">Titanic</option>
          <option value="housing">Housing</option>
        </select>
      </div>
      <div class="prop-group">
        <label>Task</label>
        <select id="am-task">
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
          <option value="clustering">Clustering</option>
        </select>
      </div>
      <div class="prop-group">
        <label>Algorithm</label>
        <select id="am-algo">
          <option value="Random Forest">Random Forest</option>
          <option value="Decision Tree">Decision Tree</option>
          <option value="Logistic Regression">Logistic Regression</option>
          <option value="KNN">KNN</option>
          <option value="Linear Regression">Linear Regression</option>
          <option value="KMeans">KMeans</option>
        </select>
      </div>
      <div style="margin-top:16px;display:flex;gap:10px;">
        <button id="am-build" style="flex:1;">Build Pipeline</button>
        <button id="am-cancel" style="flex:1;background:#555;">Cancel</button>
      </div>
    `;
    this.showModal(div);

    document.getElementById("am-cancel").addEventListener("click", () => this.closeModal());
    document.getElementById("am-build").addEventListener("click", () => {
      const dataset = document.getElementById("am-dataset").value;
      const task    = document.getElementById("am-task").value;
      const algo    = document.getElementById("am-algo").value;
      this._buildAutoModel(dataset, task, algo);
      this.closeModal();
    });
  }

  _buildAutoModel(dataset, task, algo) {
    this.newProcess();
    this.process.name = `AutoModel: ${dataset} ${task}`;

    const id1 = this.process.addOperator("Generate Data", 80, 120);
    this.process.operators[id1].params.dataset = dataset;

    if (task === "clustering") {
      const id2 = this.process.addOperator("Normalize", 280, 120);
      const id3 = this.process.addOperator(algo, 480, 120);
      const id4 = this.process.addOperator("Performance Clustering", 680, 120);
      this.process.connect(id1, "out", id2, "in");
      this.process.connect(id2, "out", id3, "training");
      this.process.connect(id3, "out", id4, "in");
    } else {
      const id2 = this.process.addOperator("Replace Missing", 280, 120);
      const id3 = this.process.addOperator("Split Data", 480, 120);
      const id4 = this.process.addOperator(algo, 680, 60);
      const id5 = this.process.addOperator("Apply Model", 880, 120);
      const perfOp = task === "classification" ? "Performance Classification" : "Performance Regression";
      const id6 = this.process.addOperator(perfOp, 1080, 120);

      this.process.connect(id1, "out", id2, "in");
      this.process.connect(id2, "out", id3, "in");
      this.process.connect(id3, "train", id4, "training");
      this.process.connect(id4, "model", id5, "model");
      this.process.connect(id3, "test", id5, "data");
      this.process.connect(id5, "out", id6, "in");
    }

    this.canvasRenderer.fitView();
    this.pushUndo();
    this.log(`AutoModel pipeline built: ${dataset} / ${task} / ${algo}`);
  }
}

/* ── Initialise on DOM ready ──────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
