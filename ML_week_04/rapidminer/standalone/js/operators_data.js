/* ═══════════════════════════════════════════════════════════════════════════
   operators_data.js – Data I/O operators: ReadCSV, ReadJSON, WriteCSV,
   Store, Retrieve, GenerateData, LoadSample.
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

/* ── Shared repository (Store / Retrieve) ─────────────────────────────────── */
const _sharedRepo = {};

/* ── Sample datasets ──────────────────────────────────────────────────────── */
const SAMPLE_DATASETS = {};

SAMPLE_DATASETS.iris = () => {
  const cols = ["sepal_length","sepal_width","petal_length","petal_width","species"];
  const raw = [
    [5.1,3.5,1.4,0.2,"setosa"],[4.9,3.0,1.4,0.2,"setosa"],[4.7,3.2,1.3,0.2,"setosa"],
    [4.6,3.1,1.5,0.2,"setosa"],[5.0,3.6,1.4,0.2,"setosa"],[5.4,3.9,1.7,0.4,"setosa"],
    [4.6,3.4,1.4,0.3,"setosa"],[5.0,3.4,1.5,0.2,"setosa"],[4.4,2.9,1.4,0.2,"setosa"],
    [4.9,3.1,1.5,0.1,"setosa"],[5.4,3.7,1.5,0.2,"setosa"],[4.8,3.4,1.6,0.2,"setosa"],
    [4.8,3.0,1.4,0.1,"setosa"],[4.3,3.0,1.1,0.1,"setosa"],[5.8,4.0,1.2,0.2,"setosa"],
    [5.7,4.4,1.5,0.4,"setosa"],[5.4,3.9,1.3,0.4,"setosa"],[5.1,3.5,1.4,0.3,"setosa"],
    [5.7,3.8,1.7,0.3,"setosa"],[5.1,3.8,1.5,0.3,"setosa"],
    [7.0,3.2,4.7,1.4,"versicolor"],[6.4,3.2,4.5,1.5,"versicolor"],[6.9,3.1,4.9,1.5,"versicolor"],
    [5.5,2.3,4.0,1.3,"versicolor"],[6.5,2.8,4.6,1.5,"versicolor"],[5.7,2.8,4.5,1.3,"versicolor"],
    [6.3,3.3,4.7,1.6,"versicolor"],[4.9,2.4,3.3,1.0,"versicolor"],[6.6,2.9,4.6,1.3,"versicolor"],
    [5.2,2.7,3.9,1.4,"versicolor"],[5.0,2.0,3.5,1.0,"versicolor"],[5.9,3.0,4.2,1.5,"versicolor"],
    [6.0,2.2,4.0,1.0,"versicolor"],[6.1,2.9,4.7,1.4,"versicolor"],[5.6,2.9,3.6,1.3,"versicolor"],
    [6.7,3.1,4.4,1.4,"versicolor"],[5.6,3.0,4.5,1.5,"versicolor"],[5.8,2.7,4.1,1.0,"versicolor"],
    [6.2,2.2,4.5,1.5,"versicolor"],[5.6,2.5,3.9,1.1,"versicolor"],
    [6.3,3.3,6.0,2.5,"virginica"],[5.8,2.7,5.1,1.9,"virginica"],[7.1,3.0,5.9,2.1,"virginica"],
    [6.3,2.9,5.6,1.8,"virginica"],[6.5,3.0,5.8,2.2,"virginica"],[7.6,3.0,6.6,2.1,"virginica"],
    [4.9,2.5,4.5,1.7,"virginica"],[7.3,2.9,6.3,1.8,"virginica"],[6.7,2.5,5.8,1.8,"virginica"],
    [7.2,3.6,6.1,2.5,"virginica"],[6.5,3.2,5.1,2.0,"virginica"],[6.4,2.7,5.3,1.9,"virginica"],
    [6.8,3.0,5.5,2.1,"virginica"],[5.7,2.5,5.0,2.0,"virginica"],[5.8,2.8,5.1,2.4,"virginica"],
    [6.4,3.2,5.3,2.3,"virginica"],[6.5,3.0,5.5,1.8,"virginica"],[7.7,3.8,6.7,2.2,"virginica"],
    [7.7,2.6,6.9,2.3,"virginica"],[6.0,2.2,5.0,1.5,"virginica"],
  ];
  const df = new DataFrame(cols, raw);
  df.roles.species = "label";
  return df;
};

SAMPLE_DATASETS.titanic = () => {
  const cols = ["pclass","sex","age","sibsp","parch","fare","embarked","survived"];
  const raw = [
    [1,"male",22,1,0,7.25,"S",0],[1,"female",38,1,0,71.28,"C",1],[3,"female",26,0,0,7.92,"S",1],
    [1,"female",35,1,0,53.10,"S",1],[3,"male",35,0,0,8.05,"S",0],[3,"male",NaN,0,0,8.46,"Q",0],
    [1,"male",54,0,0,51.86,"S",0],[3,"male",2,3,1,21.07,"S",0],[3,"female",27,0,2,11.13,"S",1],
    [2,"female",14,1,0,30.07,"C",1],[3,"female",4,1,1,16.70,"S",1],[1,"female",58,0,0,26.55,"S",1],
    [3,"male",20,0,0,8.05,"S",0],[3,"male",39,1,5,31.27,"S",0],[3,"female",14,0,0,7.85,"S",0],
    [2,"female",55,0,0,16.00,"S",1],[3,"male",2,4,1,29.12,"Q",0],[2,"male",NaN,0,0,13.00,"S",1],
    [3,"female",31,1,0,18.00,"S",0],[3,"female",NaN,0,0,7.22,"C",1],
    [2,"male",35,0,0,26.00,"S",0],[1,"male",34,0,0,13.00,"S",1],[2,"female",15,0,0,8.03,"S",0],
    [1,"male",28,0,0,35.50,"S",1],[3,"female",8,3,1,21.07,"S",0],[3,"female",38,1,5,31.39,"S",0],
    [3,"male",NaN,0,0,7.90,"S",0],[1,"male",19,3,2,263.0,"S",0],[3,"female",NaN,0,0,7.75,"Q",1],
    [3,"male",NaN,0,0,7.23,"C",0],
  ];
  const df = new DataFrame(cols, raw);
  df.roles.survived = "label";
  return df;
};

SAMPLE_DATASETS.housing = () => {
  const cols = ["rooms","area","age","distance","price"];
  const rng = _seedRng(123);
  const rows = [];
  for (let i = 0; i < 60; i++) {
    const rooms = Math.floor(rng() * 6) + 2;
    const area = 40 + Math.floor(rng() * 160);
    const age = Math.floor(rng() * 50);
    const dist = +(rng() * 30).toFixed(1);
    const price = Math.round(rooms * 20000 + area * 800 - age * 500 - dist * 1000 + (rng() - 0.5) * 20000);
    rows.push([rooms, area, age, dist, price]);
  }
  const df = new DataFrame(cols, rows);
  df.roles.price = "label";
  return df;
};

/* ── ReadCSV ──────────────────────────────────────────────────────────────── */

registerOperator("Read CSV", () => {
  const op = new Operator("Read CSV", OpCategory.DATA,
    [],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("separator", ParamKind.CHOICE, ",", "Column separator", [",", ";", "\\t", "|"]),
      new ParamSpec("header", ParamKind.BOOL, true, "First row is header"),
      new ParamSpec("_csv_text", ParamKind.STRING, "", "CSV content (injected at runtime)"),
    ],
    "Read tabular data from a CSV string."
  );
  op.execute = function(inp) {
    const text = this.params._csv_text;
    if (!text) throw new Error("Read CSV: no CSV data provided. Upload a file first.");
    const sep = this.params.separator === "\\t" ? "\t" : this.params.separator;
    return { out: DataFrame.fromCSV(text, sep, this.params.header) };
  };
  return op;
});

/* ── ReadJSON ─────────────────────────────────────────────────────────────── */

registerOperator("Read JSON", () => {
  const op = new Operator("Read JSON", OpCategory.DATA,
    [],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("_json_text", ParamKind.STRING, "", "JSON content")],
    "Read tabular data from a JSON array."
  );
  op.execute = function(inp) {
    const arr = JSON.parse(this.params._json_text || "[]");
    return { out: DataFrame.fromJSON(arr) };
  };
  return op;
});

/* ── WriteCSV ─────────────────────────────────────────────────────────────── */

registerOperator("Write CSV", () => {
  const op = new Operator("Write CSV", OpCategory.DATA,
    [new Port("in", PortType.EXAMPLE_SET)],
    [new Port("out", PortType.EXAMPLE_SET)],
    [new ParamSpec("separator", ParamKind.CHOICE, ",", "Separator", [",", ";", "\\t", "|"])],
    "Download data as CSV file."
  );
  op.execute = function(inp) {
    const df = inp.in;
    if (df) {
      const sep = this.params.separator === "\\t" ? "\t" : this.params.separator;
      const csv = df.toCSV(sep);
      // Trigger browser download
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = "output.csv"; a.click();
      URL.revokeObjectURL(url);
    }
    return { out: df };
  };
  return op;
});

/* ── Store ────────────────────────────────────────────────────────────────── */

registerOperator("Store", () => {
  const op = new Operator("Store", OpCategory.DATA,
    [new Port("in", PortType.ANY)],
    [new Port("out", PortType.ANY)],
    [new ParamSpec("name", ParamKind.STRING, "stored_data", "Repository name")],
    "Store data in shared memory."
  );
  op.execute = function(inp) {
    _sharedRepo[this.params.name] = inp.in;
    return { out: inp.in };
  };
  return op;
});

/* ── Retrieve ─────────────────────────────────────────────────────────────── */

registerOperator("Retrieve", () => {
  const op = new Operator("Retrieve", OpCategory.DATA,
    [],
    [new Port("out", PortType.ANY)],
    [new ParamSpec("name", ParamKind.STRING, "stored_data", "Repository name")],
    "Retrieve data from shared memory."
  );
  op.execute = function(inp) {
    const data = _sharedRepo[this.params.name];
    if (data == null) throw new Error(`Retrieve: '${this.params.name}' not found in repository.`);
    return { out: data };
  };
  return op;
});

/* ── Generate Data ────────────────────────────────────────────────────────── */

registerOperator("Generate Data", () => {
  const op = new Operator("Generate Data", OpCategory.DATA,
    [],
    [new Port("out", PortType.EXAMPLE_SET)],
    [
      new ParamSpec("dataset", ParamKind.CHOICE, "iris", "Sample dataset", ["iris", "titanic", "housing"]),
    ],
    "Generate a built-in sample dataset."
  );
  op.execute = function(inp) {
    const factory = SAMPLE_DATASETS[this.params.dataset];
    if (!factory) throw new Error(`Generate Data: unknown dataset '${this.params.dataset}'.`);
    return { out: factory() };
  };
  return op;
});
