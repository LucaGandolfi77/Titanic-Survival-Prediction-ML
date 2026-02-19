# ============================================================
# Makefile — Common Commands
# ============================================================
# Usage: make <target>
# ============================================================

.PHONY: help install install-dev train optimize serve test lint \
        docker-build docker-up docker-down mlflow clean

PYTHON     := python
PIP        := pip
PYTEST     := pytest
PROJECT    := titanic-mlops

# Default target
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────

install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install all dependencies (production + development)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ── ML Pipeline ──────────────────────────────────────────────

train:  ## Train all models (use MODEL=xgboost to train one)
ifdef MODEL
	$(PYTHON) train.py --model $(MODEL)
else
	$(PYTHON) train.py
endif

optimize:  ## Run Optuna hyperparameter optimization
	$(PYTHON) optimize.py

serve:  ## Start the FastAPI prediction server
	$(PYTHON) serve.py

serve-dev:  ## Start API with auto-reload for development
	$(PYTHON) serve.py --reload

# ── Quality ──────────────────────────────────────────────────

test:  ## Run all unit tests
	$(PYTEST) tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Lint code with ruff
	ruff check src/ tests/ train.py optimize.py serve.py

format:  ## Format code with black
	black src/ tests/ train.py optimize.py serve.py

# ── MLflow ───────────────────────────────────────────────────

mlflow:  ## Start MLflow tracking UI locally
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns

# ── Docker ───────────────────────────────────────────────────

docker-build:  ## Build the Docker image
	docker build -t $(PROJECT) .

docker-up:  ## Start all services (MLflow + API) via docker compose
	docker compose up -d

docker-down:  ## Stop all docker compose services
	docker compose down

# ── Housekeeping ─────────────────────────────────────────────

clean:  ## Remove generated artefacts
	rm -rf data/processed/*.pkl
	rm -rf models/*.joblib models/*.txt models/*.png
	rm -rf logs/*.log
	rm -rf mlruns/
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache htmlcov
	@echo "Cleaned ✓"
