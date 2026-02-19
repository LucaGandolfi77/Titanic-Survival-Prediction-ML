.PHONY: install test lint examples clean xor spiral regression mnist notebooks all

PYTHON ?= python3
PIP    ?= pip

# ── Setup ─────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# ── Tests ─────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Examples ──────────────────────────────────────────
xor:
	$(PYTHON) examples/xor_example.py

spiral:
	$(PYTHON) examples/spiral_example.py

regression:
	$(PYTHON) examples/regression_example.py

mnist:
	$(PYTHON) train_mnist.py

examples: xor spiral regression

# ── Clean ─────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
	rm -rf outputs/plots/*.png
	rm -rf outputs/models/*.npz

# ── All ───────────────────────────────────────────────
all: install test examples
