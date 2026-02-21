.PHONY: install test search fast train compare viz clean

install:
	pip install -e ".[dev]"

test:
	TMPDIR=/tmp python -m pytest tests/ -v --tb=short

export KMP_DUPLICATE_LIB_OK := TRUE
export TMPDIR := /tmp

search:
	python scripts/run_search.py

fast:
	python scripts/run_search.py --config configs/fast.yaml

train:
	python scripts/train_best.py outputs/best_genome.json --epochs 100

compare:
	python scripts/compare_resnet.py outputs/best_genome.json

viz:
	python scripts/visualize.py outputs/

clean:
	rm -rf outputs/architectures outputs/plots outputs/logs
	find . -type d -name __pycache__ -exec rm -rf {} +
