# Makefile for ML_week_04 benchmarking project

.PHONY: help install train benchmark dashboard clean test

help:
	@echo "=== ML Week 04 Commands ==="
	@echo "make install        - Install dependencies"
	@echo "make train          - Run training on default dataset"
	@echo "make benchmark      - Run benchmarking suite"
	@echo "make dashboard      - Generate interactive HTML dashboard"
	@echo "make test           - Run unit tests"
	@echo "make clean          - Remove outputs and cache"

install:
	python3 -m venv venv 2>/dev/null || true
	. venv/bin/activate && pip install -U pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt

train:
	. venv/bin/activate && python src/train.py --config config/experiment.yaml

benchmark:
	. venv/bin/activate && python src/benchmark.py --config config/experiment.yaml

dashboard:
	. venv/bin/activate && python src/generate_dashboard.py --results results/ --output outputs/dashboard.html

test:
	. venv/bin/activate && pytest tests/ -v

clean:
	rm -rf venv __pycache__ .pytest_cache outputs/ results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
