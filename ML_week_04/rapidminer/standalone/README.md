# RapidMiner Lite – Standalone Web App

A **zero-dependency, browser-only** clone of [RapidMiner Studio](https://rapidminer.com/) built entirely in vanilla HTML, CSS and JavaScript.  
No server, no npm, no build step — just open `index.html` in any modern browser.

---

## File Structure

```
standalone/
├── index.html        # Entry point – toolbar, 3-panel layout, modal, hidden file inputs
├── style.css         # Full dark theme (CSS custom properties), responsive layout
├── README.md         # This file
└── js/               # Modular JavaScript (15 files, loaded in dependency order)
    ├── dataframe.js        # DataFrame class – CSV/JSON parsing, column ops, stats, transforms
    ├── ml.js               # Pure-JS ML algorithms & metrics (13 models + PCA)
    ├── operators.js        # Operator base class, port/param types, registry
    ├── operators_data.js   # Data operators: Read CSV/JSON, Write, Store, Retrieve, Generate
    ├── operators_transform.js  # 17 transform operators (filter, sort, join, pivot, …)
    ├── operators_feature.js    # 9 feature engineering ops (PCA, encoding, selection, …)
    ├── operators_model.js      # 13 model operators (classification, regression, clustering)
    ├── operators_eval.js       # 6 evaluation ops (Apply Model, Cross Validation, …)
    ├── operators_viz.js        # 6 Canvas 2D visualization operators
    ├── process.js          # Process model, topological sort, ProcessRunner, sample processes
    ├── canvas.js           # Canvas renderer – nodes, Bézier wires, zoom/pan, drag-drop
    ├── palette.js          # Left panel – searchable, collapsible operator palette
    ├── properties.js       # Right panel – dynamic property inspector
    ├── results.js          # Results panel – tables, metric cards, confusion matrix, charts
    └── app_main.js         # Main controller – toolbar, tabs, undo/redo, AutoModel wizard
```

### `index.html`
- **Toolbar**: New / Open / Save process, sample process & dataset dropdowns, Run / Stop / Undo / Redo, AutoModel wizard, Clear canvas
- **Left panel**: Searchable operator palette grouped by category (Data, Transform, Feature, Model, Evaluation, Visualization, Utility)
- **Center**: Tabbed area with **Design** (canvas), **Results** (tables + metrics + charts), **Log** (execution trace)
- **Right panel**: Property inspector — shows parameters for the selected operator
- **Modal overlay**: Used for AutoModel wizard, data preview, and dialogs

### `style.css`
- Dark theme via CSS custom properties (`--bg`, `--surface`, `--accent`, …)
- Category colour coding (Data = blue, Transform = green, Feature = amber, Model = red, Evaluation = purple, Visualization = pink, Utility = grey)
- Port type colours (`--port-es`, `--port-model`, `--port-perf`, `--port-any`)
- Responsive breakpoints for narrower screens

### Engine Layer (`js/dataframe.js` → `js/process.js`)

| File | Description |
|---|---|
| **dataframe.js** | Lightweight tabular data: `columns`, `data`, `roles`. CSV/JSON parsing, select/filter/sort/join/pivot/aggregate/transpose, describe stats, correlation matrix, normalize, one-hot/label encoding, train-test split, `toMatrix()` / `toArray()` for ML |
| **ml.js** | Pure-JS implementations of 13 ML algorithms + PCA. Linear algebra helpers (`dot`, `matMul`, covariance). `Metrics` object (accuracy, precision, recall, F1, confusion matrix, MSE, RMSE, MAE, R², silhouette). `ML_MODELS` registry |
| **operators.js** | `PortType`, `OpCategory`, `ParamKind` enums. `Operator` base class with `execute()`. `Connection` class. Registry functions: `registerOperator`, `getOperator`, `listOperators`, `operatorsByCategory` |
| **operators_data.js** | 6 operators: Read CSV, Read JSON, Write CSV, Store, Retrieve, Generate Data. Includes `SAMPLE_DATASETS` (iris 60 rows, titanic 30 rows, housing 60 rows) |
| **operators_transform.js** | 17 transform operators matching the Python project |
| **operators_feature.js** | 9 feature engineering operators (PCA, variance threshold, forward/backward selection, correlation, encodings) |
| **operators_model.js** | 13 model training operators with auto feature/label preparation helpers |
| **operators_eval.js** | 6 evaluation operators: Apply Model, Performance (Classification/Regression/Clustering), Cross Validation, Feature Importance |
| **operators_viz.js** | 6 Canvas 2D visualization operators: Distribution, Scatter, Box Plot, Heatmap, ROC Curve, Parallel Coordinates |
| **process.js** | `Process` class (DAG of operators + connections), Kahn's topological sort, `ProcessRunner` with callbacks, 2 sample processes |

### GUI Layer (`js/canvas.js` → `js/app_main.js`)

| File | Description |
|---|---|
| **canvas.js** | HTML5 Canvas renderer: operator nodes (rounded rect + category colour header + ports), Bézier wires, grid. Zoom/pan (wheel + drag), drag-drop from palette, click-to-connect ports, double-click to edit, hit-testing |
| **palette.js** | Left panel: reads operator registry, groups by category, renders collapsible sections with search filtering and drag handles |
| **properties.js** | Right panel: dynamically generates form controls for selected operator's params (text, number, bool, choice, column picker). Shows category badge, port info, delete button |
| **results.js** | Results tab: renders DataFrames as scrollable tables, performance metrics as cards, confusion matrices, chart canvases from visualization operators |
| **app_main.js** | Main controller: toolbar button bindings, tab switching, file I/O (open/save `.rmp.json`, import CSV), undo/redo (50-deep JSON stack), keyboard shortcuts, AutoModel wizard modal |

---

## Operator Catalogue

All operators mirror the Python RapidMiner Lite project:

### Data (7)
Read CSV, Read JSON, Write CSV, Store, Retrieve, Generate Data, Load Sample

### Transform (17)
Select Attributes, Filter Examples, Rename, Set Role, Replace Missing, Normalize, Discretize, Generate Attributes, Remove Duplicates, Sort, Sample, Split Data, Append, Join, Aggregate, Transpose, Pivot

### Feature Engineering (9)
PCA, Variance Threshold, Forward Selection, Backward Elimination, Correlation Matrix, Weight by Correlation, One Hot Encoding, Label Encoding, Target Encoding

### Models (13)
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, Linear Regression, Ridge, Lasso, KMeans, DBSCAN, Agglomerative

### Evaluation (6)
Apply Model, Performance (Classification), Performance (Regression), Performance (Clustering), Cross Validation, Feature Importance

### Visualization (6)
Data Distribution, Scatter Plot, Box Plot, Correlation Heatmap, ROC Curve, Parallel Coordinates

### Utility (3)
Log to Console, Set Macro, Branch

---

## How to Use

1. **Open** `index.html` in a browser (Chrome, Firefox, Edge, Safari)
2. **Drag** operators from the left palette onto the canvas
3. **Connect** output ports → input ports by clicking and dragging
4. **Configure** parameters in the right panel
5. **Run** the process with ▶ and view results in the Results / Log tabs
6. **Save** the process as a `.rmp` JSON file for later

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Delete` / `Backspace` | Remove selected operator or wire |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+A` | Select all operators |
| `Escape` | Deselect / close modal |
| Mouse wheel | Zoom in/out |
| Middle-click drag | Pan canvas |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        index.html                             │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │ Palette  │  │ Canvas + Tabs    │  │  Properties       │   │
│  │ (left)   │  │   (center)       │  │   (right)         │   │
│  └────┬─────┘  └───────┬──────────┘  └──────┬────────────┘   │
│       │                │                     │                │
│       └────────────────┼─────────────────────┘                │
│           GUI Layer    │                                      │
│  canvas.js  palette.js │  properties.js  results.js           │
│                 app_main.js                                   │
└────────────────────────┼─────────────────────────────────────┘
                         │
           ┌─────────────▼─────────────┐
           │      Engine Layer         │
           │                           │
           │  dataframe.js   ml.js     │
           │  operators.js             │
           │  operators_data.js        │
           │  operators_transform.js   │
           │  operators_feature.js     │
           │  operators_model.js       │
           │  operators_eval.js        │
           │  operators_viz.js         │
           │  process.js               │
           └───────────────────────────┘
```

**No external dependencies.** All ML algorithms (logistic regression, decision trees, random forests, KNN, K-Means, PCA, etc.) are implemented from scratch in pure JavaScript.

---

## Comparison with Python Version

| Feature | Python (tkinter) | Standalone (HTML/CSS/JS) |
|---|---|---|
| Runtime | Python 3.10+ | Any modern browser |
| ML backend | scikit-learn | Pure JS implementations |
| Visualisation | matplotlib / seaborn | Canvas 2D API |
| Process format | `.rmp` JSON | `.rmp` JSON (compatible) |
| Operator count | 61 | 61 |
| Installation | `pip install -r requirements.txt` | None — open `index.html` |
| File I/O | Filesystem | Browser FileReader + download |

---

## License

Educational project. See repository root for license details.
