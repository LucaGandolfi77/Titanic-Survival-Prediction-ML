# ============================================================================
#  Makefile — Lightweight Mobile Object Tracker
# ============================================================================

PYTHON   ?= python3
VENV     ?= .venv
PIP      := $(VENV)/bin/pip
PY       := $(VENV)/bin/python

# ── Setup ─────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "\n✅  Environment ready.  Activate with:  source $(VENV)/bin/activate"

# ── Data preparation ──────────────────────────────────────────────────────
.PHONY: data
data:
	$(PY) scripts/prepare_data.py

# ── Training ──────────────────────────────────────────────────────────────
.PHONY: train
train:
	$(PY) -m src.training.train_siamese --config configs/training_config.yaml

# ── Export to TFLite ──────────────────────────────────────────────────────
.PHONY: export
export:
	$(PY) -m src.training.export_tflite --quant fp16

# ── Desktop demo ──────────────────────────────────────────────────────────
.PHONY: demo
demo:
	$(PY) -m src.demo.desktop_demo

# ── Benchmark ─────────────────────────────────────────────────────────────
.PHONY: benchmark
benchmark:
	$(PY) scripts/benchmark_tflite_m1.py

# ── CoreML conversion (optional) ─────────────────────────────────────────
.PHONY: coreml
coreml:
	$(PY) scripts/convert_to_coreml.py

# ── Clean ─────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(VENV) __pycache__ src/__pycache__ logs/tensorboard
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
