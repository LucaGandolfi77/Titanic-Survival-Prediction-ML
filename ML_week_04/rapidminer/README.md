# RapidMiner Lite

A fully‑featured, open‑source **desktop data‑science workbench** inspired by [RapidMiner Studio](https://rapidminer.com/).  
Built entirely in Python with **tkinter**, it provides a visual drag‑and‑drop canvas for building end‑to‑end machine‑learning pipelines — no code required.

---

## Highlights

| Feature | Details |
|---|---|
| **61 built‑in operators** | Data I/O, transformation, feature engineering, classification, regression, clustering, evaluation, and visualization |
| **Visual process designer** | Dark‑themed canvas with Bézier wires, zoom, pan, undo/redo |
| **DAG execution engine** | Topological sort (Kahn's algorithm), async execution with live progress callbacks |
| **AutoModel wizard** | One‑click model comparison across multiple algorithms |
| **Process serialisation** | JSON `.rmp` files — save, load, export, share |
| **Repository** | Persist ExampleSets, models, and results to disk (pickle) |
| **Comprehensive test suite** | 116 pytest tests covering every engine module |

---

## Project Structure

```
rapidminer/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
│
├── engine/                      # Core engine (no GUI dependency)
│   ├── operator_base.py         # Operator ABC, Port, Registry, Connection
│   ├── operators_data.py        # Read CSV/Excel/JSON/DB, Write CSV, Store, Retrieve
│   ├── operators_transform.py   # 17 transform operators
│   ├── operators_feature.py     # PCA, Variance Threshold, Encoding, ...
│   ├── operators_model.py       # 13 ML models (classification, regression, clustering)
│   ├── operators_eval.py        # Apply Model, Performance metrics, Cross Validation
│   ├── operators_viz.py         # 6 visualization operators (matplotlib/seaborn)
│   ├── process_runner.py        # Process, topological sort, ProcessRunner
│   └── repository.py            # Save/load processes and results to disk
│
├── gui/                         # tkinter GUI layer
│   ├── theme.py                 # Dark theme constants
│   ├── wire.py                  # Bézier wire rendering
│   ├── operator_node.py         # Operator node drawing
│   ├── canvas.py                # ProcessCanvas (drag, drop, wire, zoom, pan)
│   ├── left_panel.py            # Operator palette + process tree
│   ├── right_panel.py           # Parameter inspector
│   ├── results_panel.py         # Execution results viewer
│   └── automodel_wizard.py      # AutoModel comparison wizard
│
├── sample_data/                 # Bundled datasets
│   ├── iris.csv                 # Classic Iris (150 rows)
│   ├── titanic.csv              # Titanic passengers (30 rows)
│   ├── housing.csv              # Housing prices (40 rows)
│   ├── churn.csv                # Customer churn (30 rows)
│   ├── weather.csv              # Weather observations (30 rows)
│   └── products.csv             # Product catalog (25 rows)
│
├── sample_processes/            # Ready‑to‑run .rmp process files
│   ├── iris_classification.rmp
│   ├── iris_clustering.rmp
│   └── iris_eda.rmp
│
└── tests/                       # pytest test suite (116 tests)
    ├── conftest.py              # Shared fixtures
    ├── test_operator_base.py
    ├── test_operators_data.py
    ├── test_operators_transform.py
    ├── test_operators_feature.py
    ├── test_operators_model.py
    ├── test_operators_eval.py
    ├── test_operators_viz.py
    ├── test_process_runner.py
    └── test_repository.py
```

---

## Requirements

- **Python 3.10+**
- **tkinter** (ships with most Python distributions)
- Runtime dependencies listed in `requirements.txt`:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-user>/rapidminer-lite.git
cd rapidminer-lite

# (Optional) Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Launch the GUI

```bash
python main.py
```

The application opens a dark‑themed window with three panels:

| Panel | Purpose |
|---|---|
| **Left** | Operator palette (drag operators onto the canvas) and process tree |
| **Center** | Visual canvas — connect operator ports with wires |
| **Right** | Parameter inspector for the selected operator |

### Run a sample process

1. **File → Open Sample Process → iris_classification.rmp**
2. Click **▶ Run** in the toolbar.
3. Switch to the **Results** tab to inspect predictions, performance metrics, and charts.

### Build a process from scratch

1. Drag a **Read CSV** operator onto the canvas.
2. Set the `filepath` parameter to one of the sample datasets (e.g. `sample_data/iris.csv`).
3. Drag a **Set Role** operator and connect `Read CSV → out` to `Set Role → in`.
4. Set `column = species`, `role = label`.
5. Add a **Random Forest** operator and wire `Set Role → out` to `Random Forest → training`.
6. Add an **Apply Model** operator — connect the model and data ports.
7. Add **Performance (Classification)** to view accuracy, F1, confusion matrix, and ROC AUC.
8. Click **▶ Run**.

---

## Operator Reference

### Data (7 operators)

| Operator | Description |
|---|---|
| Read CSV | Load a CSV file with configurable separator, header, encoding |
| Read Excel | Load an Excel workbook (`.xlsx`) |
| Read JSON | Load a JSON file (records / columns orient) |
| Read Database | Query a SQLite database |
| Write CSV | Export an ExampleSet to CSV |
| Store | Save an ExampleSet to the in‑memory repository |
| Retrieve | Load an ExampleSet from the in‑memory repository |

### Transform (17 operators)

| Operator | Description |
|---|---|
| Select Attributes | Keep or remove selected columns |
| Filter Examples | Filter rows with a pandas query expression |
| Rename | Rename columns (`old=new` mapping) |
| Set Role | Assign a role (label, id, weight, feature) to a column |
| Replace Missing | Impute missing values (mean, median, mode, constant) |
| Normalize | Z‑score, min‑max, or log normalization |
| Discretize | Bin a numeric column into intervals |
| Generate Attributes | Create new columns with a formula (`df.eval`) |
| Remove Duplicates | Drop duplicate rows |
| Sort | Sort by a column (ascending / descending) |
| Sample | Random sample by fraction or absolute count |
| Split Data | Train/test split with optional stratification |
| Append | Concatenate two ExampleSets vertically |
| Join | Merge two ExampleSets on key columns |
| Aggregate | Group‑by aggregation (mean, sum, count, …) |
| Transpose | Transpose rows ↔ columns |
| Pivot | Pivot table with index, columns, values, aggfunc |

### Feature Engineering (9 operators)

| Operator | Description |
|---|---|
| PCA | Principal Component Analysis |
| Variance Threshold | Remove low‑variance features |
| Forward Selection | Greedy forward feature selection (CV‑based) |
| Backward Elimination | Backward feature elimination |
| Correlation Matrix | Compute pairwise correlation matrix |
| Weight by Correlation | Rank features by absolute correlation with the label |
| One Hot Encoding | One‑hot encode categorical columns |
| Label Encoding | Integer‑encode categorical columns |
| Target Encoding | Mean‑target encode categorical columns |

### Models (13 operators)

| Operator | Description |
|---|---|
| Logistic Regression | Binary / multiclass logistic regression |
| Decision Tree | Decision tree classifier |
| Random Forest | Random forest classifier |
| Gradient Boosting | Gradient boosting classifier |
| SVM | Support Vector Machine |
| KNN | K‑Nearest Neighbors |
| Naive Bayes | Gaussian Naive Bayes |
| Linear Regression | Ordinary least squares regression |
| Ridge | Ridge regression (L2) |
| Lasso | Lasso regression (L1) |
| KMeans | K‑Means clustering |
| DBSCAN | Density‑based clustering |
| Agglomerative | Hierarchical agglomerative clustering |

### Evaluation (6 operators)

| Operator | Description |
|---|---|
| Apply Model | Apply a trained model to produce predictions |
| Performance (Classification) | Accuracy, precision, recall, F1, AUC, confusion matrix |
| Performance (Regression) | MAE, MSE, RMSE, R² |
| Performance (Clustering) | Silhouette score, inertia, cluster distribution |
| Cross Validation | K‑fold cross‑validation with selectable estimator & scoring |
| Feature Importance | Extract and rank feature importances |

### Visualization (6 operators)

| Operator | Description |
|---|---|
| Data Distribution | Histogram + KDE |
| Scatter Plot | 2D scatter with optional colour / size mapping |
| Box Plot | Box plot by category |
| Correlation Heatmap | Seaborn heatmap of the correlation matrix |
| ROC Curve | Receiver Operating Characteristic curve |
| Parallel Coordinates | Parallel coordinates coloured by label / cluster |

### Utility (3 operators)

| Operator | Description |
|---|---|
| Log to Console | Print ExampleSet summary to the execution log |
| Set Macro | Store a global key‑value macro |
| Branch | Conditional routing based on a macro expression |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                     GUI Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Left     │ │ Canvas   │ │ Right / Results  │ │
│  │ Panel    │ │ (Bézier  │ │ Panel            │ │
│  │          │ │  wires)  │ │                  │ │
│  └──────────┘ └────┬─────┘ └──────────────────┘ │
└────────────────────┼────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │     Engine Layer      │
         │                       │
         │  Process  ─► Runner   │
         │    │         (topo    │
         │    │          sort)   │
         │    ▼                  │
         │  Operators            │
         │  (61 registered       │
         │   via @decorator)     │
         │                       │
         │  Repository           │
         │  (JSON + pickle)      │
         └───────────────────────┘
```

**Key design decisions:**

- **Decorator registry**: Each operator class is registered with `@register_operator`, populating a global `_OPERATOR_REGISTRY` dict. This allows dynamic discovery and instantiation by name.
- **Port system**: Typed ports (`EXAMPLE_SET`, `MODEL`, `PERFORMANCE`, `ANY`) prevent invalid connections in the GUI.
- **DAG execution**: The `ProcessRunner` builds a directed acyclic graph from connections, performs Kahn's topological sort, and executes operators sequentially, passing outputs through wires.
- **Role metadata**: Label, ID, and weight roles are stored in `df.attrs["_roles"]`, propagated through the pipeline.
- **Separation of concerns**: The `engine/` package has **zero GUI dependencies** — it can be used as a standalone library or scripted from a notebook.

---

## Running Tests

```bash
# Run all 116 tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_operators_model.py -v

# Run with coverage
python -m pytest tests/ --cov=engine --cov-report=term-missing
```

---

## Sample Datasets

| File | Rows | Columns | Description |
|---|---|---|---|
| `iris.csv` | 150 | 5 | Classic Iris flower measurements |
| `titanic.csv` | 30 | 12 | Titanic passenger survival data |
| `housing.csv` | 40 | 10 | Housing prices with features |
| `churn.csv` | 30 | 12 | Telecom customer churn |
| `weather.csv` | 30 | 7 | Daily weather observations |
| `products.csv` | 25 | 10 | Product catalog with ratings |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Write tests for new operators or functionality.
4. Ensure all tests pass: `python -m pytest tests/ -v`.
5. Submit a pull request.

---

## License

This project is provided for educational purposes. See the repository root for license details.
