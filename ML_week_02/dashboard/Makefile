.PHONY: install generate run test lint clean help

PYTHON ?= python3
PIP ?= pip
STREAMLIT ?= streamlit
PORT ?= 8502

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	$(PIP) install -r requirements.txt

generate:  ## Generate sample datasets and pre-trained models
	$(PYTHON) _generate.py

run:  ## Start the Streamlit dashboard
	$(STREAMLIT) run app.py --server.port $(PORT) --server.headless true

test:  ## Run tests with pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Lint with ruff (if installed)
	$(PYTHON) -m ruff check src/ tests/ app.py pages/ || true

clean:  ## Remove generated artifacts
	rm -rf outputs/reports/*.html outputs/reports/*.txt
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache

all: install generate test run  ## Full setup: install, generate, test, run
