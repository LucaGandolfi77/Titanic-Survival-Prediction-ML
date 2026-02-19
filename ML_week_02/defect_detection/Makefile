.PHONY: install data train export api ui test benchmark lint clean all

PYTHON  ?= python3
PIP     ?= pip
DEVICE  ?= cpu
MODEL   ?= s
EPOCHS  ?= 100
BATCH   ?= 16
PORT_API ?= 8000
PORT_UI  ?= 8503

# ──────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-cov black ruff

# ──────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────
data:
	bash scripts/download_dataset.sh

# ──────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────
train:
	bash scripts/train.sh $(MODEL) $(EPOCHS) $(BATCH) $(DEVICE)

# ──────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────
export:
	bash scripts/export_model.sh models/trained/best.pt onnx

# ──────────────────────────────────────────────────
# Serve
# ──────────────────────────────────────────────────
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT_API) --reload

ui:
	streamlit run src/ui/streamlit_app.py --server.port $(PORT_UI) --server.headless true

gradio:
	$(PYTHON) -m src.ui.gradio_app

# ──────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short -q

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing -v

# ──────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────
benchmark:
	$(PYTHON) scripts/benchmark.py --device $(DEVICE)

# ──────────────────────────────────────────────────
# Lint / Format
# ──────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# ──────────────────────────────────────────────────
# Clean
# ──────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf runs/ outputs/ .ruff_cache/

# ──────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────
all: install data train test
	@echo "✓ Full pipeline complete"
