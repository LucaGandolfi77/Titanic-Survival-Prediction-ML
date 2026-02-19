# ─────────────────────────────────────────────────
# ML Playground – Makefile
# ─────────────────────────────────────────────────
.PHONY: install run test lint clean help

PYTHON   := python3
PIP      := $(PYTHON) -m pip
STREAMLIT:= $(PYTHON) -m streamlit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: ## Launch the Streamlit app
	$(STREAMLIT) run app.py --server.port 8501

test: ## Run pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Lint with ruff (if installed)
	$(PYTHON) -m ruff check src/ pages/ app.py --fix || true

clean: ## Remove caches and generated artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/models/* outputs/predictions/* outputs/reports/*
