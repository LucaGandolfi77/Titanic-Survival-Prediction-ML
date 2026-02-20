.PHONY: install install-dev test test-cov demo-titanic demo-wine run-custom mlflow-ui clean all

PYTHON ?= python
TARGET ?= Survived
CSV    ?= data/sample/titanic.csv

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=automl --cov-report=html --cov-report=term-missing

demo-titanic:
	$(PYTHON) examples/binary_classification.py

demo-wine:
	$(PYTHON) examples/multiclass.py

demo-regression:
	$(PYTHON) examples/regression.py

run-custom:
	$(PYTHON) -m automl fit --csv $(CSV) --target $(TARGET)

mlflow-ui:
	mlflow ui --backend-store-uri ./outputs/mlruns --port 5000

clean:
	rm -rf outputs/models/* outputs/reports/* outputs/generated_apis/*
	rm -rf .pytest_cache dist build *.egg-info htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install test demo-titanic
