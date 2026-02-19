.PHONY: install train-cartpole train-ddqn train-thermal \
       evaluate compare test tensorboard clean help

PYTHON ?= python

# ══════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════
install:  ## Install dependencies
	pip install -r requirements.txt

# ══════════════════════════════════════════════════════════
# Training — Stage 1
# ══════════════════════════════════════════════════════════
train-cartpole:  ## Train vanilla DQN on CartPole-v1
	$(PYTHON) train.py --config config/cartpole_dqn.yaml

# ══════════════════════════════════════════════════════════
# Training — Stage 2
# ══════════════════════════════════════════════════════════
train-ddqn:  ## Train DDQN + PER on CartPole-v1
	$(PYTHON) train.py --config config/cartpole_ddqn.yaml

train-dueling:  ## Train Dueling DQN on CartPole-v1
	$(PYTHON) train.py --config config/cartpole_ddqn.yaml \
		--config config/cartpole_ddqn.yaml

# ══════════════════════════════════════════════════════════
# Training — Stage 3
# ══════════════════════════════════════════════════════════
train-thermal:  ## Train DDQN on Thermal Control environment
	$(PYTHON) train.py --config config/thermal_control.yaml

# ══════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════
evaluate:  ## Evaluate a trained agent (pass CKPT= and CFG=)
	$(PYTHON) evaluate.py --config $(CFG) --checkpoint $(CKPT) --episodes 50

compare:  ## Compare DQN vs PID on thermal control
	$(PYTHON) compare_controllers.py \
		--config config/thermal_control.yaml \
		--checkpoint outputs/models/thermal_control/checkpoint_best.pt

# ══════════════════════════════════════════════════════════
# Testing
# ══════════════════════════════════════════════════════════
test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-agents:  ## Run agent tests only
	$(PYTHON) -m pytest tests/test_agents.py -v

test-memory:  ## Run memory buffer tests only
	$(PYTHON) -m pytest tests/test_memory.py -v

test-env:  ## Run thermal environment tests only
	$(PYTHON) -m pytest tests/test_thermal_env.py -v

# ══════════════════════════════════════════════════════════
# Monitoring
# ══════════════════════════════════════════════════════════
tensorboard:  ## Launch TensorBoard
	tensorboard --logdir outputs/tensorboard --port 6006

# ══════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════
clean:  ## Remove output artefacts
	rm -rf outputs/models/*/*.pt
	rm -rf outputs/tensorboard/*
	rm -rf outputs/plots/*.png
	rm -rf outputs/videos/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
