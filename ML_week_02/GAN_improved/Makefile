# ============================================================================
# GAN Improved — Makefile
# ============================================================================

.PHONY: help install train-vanilla train-dcgan train-cgan evaluate generate
.PHONY: test lint clean tensorboard

PYTHON := python
CONFIG_DIR := config

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────

install:  ## Install dependencies
	pip install -r requirements.txt

download-inception:  ## Pre-download InceptionV3 weights
	$(PYTHON) scripts/download_inception.py

# ── Training ─────────────────────────────────────────────────────────────

train-vanilla:  ## Train Vanilla GAN (Stage 1)
	$(PYTHON) train.py --config $(CONFIG_DIR)/vanilla_gan.yaml

train-dcgan:  ## Train DCGAN (Stage 2)
	$(PYTHON) train.py --config $(CONFIG_DIR)/dcgan.yaml

train-cgan:  ## Train Conditional GAN (Stage 3)
	$(PYTHON) train.py --config $(CONFIG_DIR)/conditional_gan.yaml

train-all: train-vanilla train-dcgan train-cgan  ## Train all stages sequentially

# ── Evaluation ───────────────────────────────────────────────────────────

evaluate-vanilla:  ## Evaluate Vanilla GAN
	$(PYTHON) evaluate.py --config $(CONFIG_DIR)/vanilla_gan.yaml \
		--checkpoint outputs/models/vanilla_gan/checkpoint_final.pt

evaluate-dcgan:  ## Evaluate DCGAN
	$(PYTHON) evaluate.py --config $(CONFIG_DIR)/dcgan.yaml \
		--checkpoint outputs/models/dcgan/checkpoint_final.pt

evaluate-cgan:  ## Evaluate Conditional GAN
	$(PYTHON) evaluate.py --config $(CONFIG_DIR)/conditional_gan.yaml \
		--checkpoint outputs/models/conditional_gan/checkpoint_final.pt

# ── Generation ───────────────────────────────────────────────────────────

generate:  ## Generate samples (default: DCGAN)
	$(PYTHON) generate.py --config $(CONFIG_DIR)/dcgan.yaml \
		--checkpoint outputs/models/dcgan/checkpoint_final.pt --n 64

generate-interpolation:  ## Generate interpolation visualization
	$(PYTHON) generate.py --config $(CONFIG_DIR)/dcgan.yaml \
		--checkpoint outputs/models/dcgan/checkpoint_final.pt \
		--interpolate --n-pairs 5 --n-steps 12

generate-conditional:  ## Generate class-conditioned grid
	$(PYTHON) generate.py --config $(CONFIG_DIR)/conditional_gan.yaml \
		--checkpoint outputs/models/conditional_gan/checkpoint_final.pt \
		--all-classes --n 80

# ── FID Statistics ───────────────────────────────────────────────────────

fid-stats:  ## Precompute FID statistics for real data
	$(PYTHON) scripts/compute_fid_statistics.py --config $(CONFIG_DIR)/dcgan.yaml

# ── Testing ──────────────────────────────────────────────────────────────

test:  ## Run all tests
	pytest tests/ -v --tb=short

test-models:  ## Run model tests only
	pytest tests/test_models.py -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Monitoring ───────────────────────────────────────────────────────────

tensorboard:  ## Launch TensorBoard
	tensorboard --logdir outputs/tensorboard --port 6006

# ── Cleanup ──────────────────────────────────────────────────────────────

clean:  ## Remove generated outputs
	rm -rf outputs/models/*/checkpoint_epoch_*.pt
	rm -rf outputs/samples/*.png
	rm -rf outputs/interpolations/*.png
	rm -rf outputs/tensorboard/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove everything including datasets
	rm -rf data/datasets/*
	rm -rf outputs/

# ── Docker ───────────────────────────────────────────────────────────────

docker-build:  ## Build Docker image
	docker build -t gan-improved .

docker-train:  ## Train inside Docker
	docker run --rm -v $(PWD)/outputs:/app/outputs gan-improved
