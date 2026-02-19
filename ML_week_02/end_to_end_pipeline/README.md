# Titanic MLOps — End-to-End ML Pipeline

> **A production-ready ML pipeline** that goes from raw data to a deployed
> prediction API — demonstrating the practices that separate an ML Engineer
> from a Data Scientist.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE OVERVIEW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Raw CSV  │───▶│ Preprocess│───▶│  Train   │───▶│  MLflow  │ │
│  │  (data/)  │    │ (sklearn) │    │ (models) │    │ Registry │ │
│  └──────────┘    └───────────┘    └──────────┘    └────┬─────┘ │
│                                        │               │       │
│                                   ┌────▼────┐    ┌─────▼─────┐ │
│                                   │ Optuna  │    │  FastAPI   │ │
│                                   │  HPO    │    │   /predict │ │
│                                   └─────────┘    └───────────┘ │
│                                                                 │
│  Config ← YAML  │  Tests ← pytest  │  Deploy ← Docker         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Area                     | Technology                        |
| ------------------------ | --------------------------------- |
| Experiment Tracking      | MLflow                            |
| Hyperparameter Tuning    | Optuna + MLflow                   |
| Model Registry           | MLflow Model Registry             |
| REST API                 | FastAPI + Pydantic v2             |
| Configuration            | YAML (`config/config.yaml`)       |
| Testing                  | pytest (preprocessing, API, predictor) |
| Containerisation         | Docker + docker compose           |
| Documentation            | OpenAPI auto-docs + this README   |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- (Optional) Docker & docker compose

### 2. Setup

```bash
# Clone and navigate
cd ML_week_02/end_to_end_pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Place Your Data

Copy `train.csv` and `test.csv` into `data/`:

```bash
cp /path/to/titanic/train.csv data/
cp /path/to/titanic/test.csv  data/
```

### 4. Start MLflow (optional — needed for experiment tracking)

```bash
make mlflow
# → MLflow UI at http://localhost:5000
```

### 5. Train All Models

```bash
make train
# or train a single model:
make train MODEL=xgboost
```

This will:
1. Load & preprocess data → `data/processed/`
2. Train Logistic Regression, Random Forest, XGBoost
3. Log all runs + models to MLflow
4. Save models locally to `models/`

### 6. Hyperparameter Optimisation

```bash
make optimize
# or with overrides:
python optimize.py --n-trials 100 --timeout 600
```

50 Optuna trials (default) are logged as nested MLflow runs.

### 7. Serve the API

```bash
make serve
# → API at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### 8. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Name": "Braund, Mr. Owen Harris"
  }'
```

Response:
```json
{"survived": 0, "probability": 0.1234}
```

---

## Testing

```bash
make test          # Run all tests
make test-cov      # With coverage report
```

---

## Docker Deployment

```bash
# Build the image
make docker-build

# Start MLflow + API
make docker-up

# Stop
make docker-down
```

---

## Project Structure

```
end_to_end_pipeline/
├── config/
│   ├── config.yaml              # All hyperparams, paths, constants
│   └── logging_config.yaml      # Python logging configuration
├── data/
│   ├── train.csv                # Raw training data
│   ├── test.csv                 # Raw test data
│   └── processed/               # Auto-generated at runtime
├── src/
│   ├── data/
│   │   └── preprocessing.py     # Feature engineering + sklearn pipeline
│   ├── models/
│   │   ├── trainer.py           # Multi-model training orchestration
│   │   └── evaluator.py         # Metrics + confusion matrix plots
│   ├── optimization/
│   │   └── optuna_optimizer.py  # Bayesian HPO → MLflow logging
│   ├── serving/
│   │   ├── api.py               # FastAPI application
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── predictor.py         # Inference wrapper
│   └── utils/
│       ├── config_loader.py     # YAML → config dict with pathlib.Path
│       └── mlflow_utils.py      # MLflow helpers
├── tests/
│   ├── test_preprocessing.py    # Feature engineering tests
│   ├── test_predictor.py        # Inference wrapper tests
│   └── test_api.py              # API endpoint tests
├── notebooks/
│   └── experiment_analysis.ipynb
├── train.py                     # CLI: python train.py
├── optimize.py                  # CLI: python optimize.py
├── serve.py                     # CLI: python serve.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

---

## Configuration

All settings live in [`config/config.yaml`](config/config.yaml).  
Python code reads it via `src.utils.config_loader.load_config()` — **zero
hard-coded values** in the codebase.

Key sections:
- `project` — name, version, seed
- `paths` — data, models, logs directories
- `mlflow` — tracking URI, experiment name, registry
- `preprocessing` — features, target, split ratio
- `training` — model hyperparameters, CV folds
- `optuna` — trial count, timeout, search space
- `serving` — host, port, log level

---

## API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Path              | Description                     |
| ------ | ----------------- | ------------------------------- |
| GET    | `/health`         | Liveness / readiness probe      |
| POST   | `/predict`        | Single passenger prediction     |
| POST   | `/predict/batch`  | Batch prediction                |
| GET    | `/model/info`     | Model metadata                  |

---

## License

MIT — Educational project for MLOps demonstration.
