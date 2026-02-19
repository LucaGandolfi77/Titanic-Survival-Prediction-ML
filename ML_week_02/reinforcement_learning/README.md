# Reinforcement Learning — DQN for Control Systems

Production-ready Reinforcement Learning project demonstrating the full
progression from classical DQN to domain-specific industrial control.

## Overview

| Stage | Environment | Agent | Key Techniques |
|-------|-------------|-------|----------------|
| **1** | CartPole-v1 | Vanilla DQN | Experience Replay, Target Network, ε-greedy |
| **2** | CartPole-v1 | DDQN / Dueling | Double DQN, Dueling architecture, PER |
| **3** | ThermalControl-v0 | DDQN + PER | Custom env, PID baseline, reward shaping |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1 — Vanilla DQN on CartPole
python train.py --config config/cartpole_dqn.yaml

# Stage 2 — DDQN with Prioritized Experience Replay
python train.py --config config/cartpole_ddqn.yaml

# Stage 3 — Thermal Control (custom environment)
python train.py --config config/thermal_control.yaml

# Evaluate a trained agent
python evaluate.py --config config/cartpole_dqn.yaml \
    --checkpoint outputs/models/cartpole/checkpoint_best.pt

# Compare DQN vs PID controller
python compare_controllers.py \
    --config config/thermal_control.yaml \
    --checkpoint outputs/models/thermal_control/checkpoint_best.pt
```

## Project Structure

```
reinforcement_learning/
├── config/                          # YAML experiment configs
│   ├── cartpole_dqn.yaml           # Stage 1 — Vanilla DQN
│   ├── cartpole_ddqn.yaml          # Stage 2 — DQN variants
│   └── thermal_control.yaml        # Stage 3 — Custom environment
├── src/
│   ├── agents/                      # DQN agent variants
│   │   ├── base_agent.py           # Abstract base class
│   │   ├── dqn_agent.py            # Vanilla DQN
│   │   ├── ddqn_agent.py           # Double DQN
│   │   └── dueling_dqn_agent.py    # Dueling DQN
│   ├── networks/                    # Neural network architectures
│   │   ├── dqn_network.py          # Standard MLP Q-network
│   │   └── dueling_network.py      # Dueling V+A decomposition
│   ├── memory/                      # Experience replay
│   │   ├── replay_buffer.py        # Uniform sampling
│   │   └── prioritized_buffer.py   # Sum-tree PER
│   ├── environments/                # Gymnasium environments
│   │   ├── thermal_control_env.py  # Custom thermal regulation
│   │   └── wrappers.py             # Reward scaling, stats, normalisation
│   ├── training/                    # Training & evaluation loops
│   │   ├── trainer.py              # Main training orchestration
│   │   └── evaluator.py            # Greedy policy evaluation
│   ├── control/                     # Classical control baselines
│   │   ├── pid_controller.py       # Discrete PID controller
│   │   └── control_utils.py        # Settling time, overshoot, IAE
│   └── utils/                       # Infrastructure
│       ├── config_loader.py        # YAML + device auto-detection
│       ├── logger.py               # TensorBoard + file logging
│       └── plotting.py             # Training curves, comparisons
├── notebooks/
│   ├── 01_cartpole_analysis.ipynb
│   ├── 02_dqn_variants_comparison.ipynb
│   └── 03_thermal_control_demo.ipynb
├── tests/
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_thermal_env.py
├── train.py                         # CLI training entry point
├── evaluate.py                      # CLI evaluation
├── compare_controllers.py           # DQN vs PID comparison
├── Makefile
├── requirements.txt
└── README.md
```

## DQN Algorithm

### Vanilla DQN (Mnih et al., 2015)

The core DQN loss minimises the temporal-difference error:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$$

where $\theta^-$ are the frozen target-network parameters.

### Double DQN (van Hasselt et al., 2016)

Decouples action **selection** from **evaluation** to reduce overestimation:

$$y = r + \gamma \, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$$

### Dueling DQN (Wang et al., 2016)

Decomposes Q-values into state-value and advantage:

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

### Prioritized Experience Replay (Schaul et al., 2016)

Samples transitions proportional to TD-error magnitude:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

## Custom Environment: Thermal Control

Simulates an avionics electronic box with lumped-capacitance thermal model:

$$C \cdot \frac{dT}{dt} = Q_{\text{gen}}(t) - \frac{T - T_{\text{amb}}}{R} - Q_{\text{fan}}(a)$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $C$ | 50 J/°C | Thermal mass |
| $R$ | 1.5 °C/W | Thermal resistance |
| $Q_{\text{gen}}$ | 30 ± 15 W | Heat generation (sinusoidal workload) |
| $T_{\text{target}}$ | 55°C | Desired operating point |
| $T_{\text{critical}}$ | 85°C | Shutdown threshold |
| Fan levels | 0–4 | Discrete cooling power [0, 5, 15, 30, 50] W |

### Reward Shaping

| Component | Value | Condition |
|-----------|-------|-----------|
| In-band bonus | +1.0 | $|T - T_{\text{target}}| \leq 5°C$ |
| Deviation penalty | −0.1/°C | Outside safe band |
| Energy penalty | −0.05 × cost | Higher fan = more penalty |
| Smoothness bonus | +0.1 | Action change ≤ 1 level |
| Critical penalty | −50 | Temperature ≥ 85°C |

## Device Support

| Device | Status |
|--------|--------|
| Apple Silicon (MPS) | ✅ Primary target |
| NVIDIA CUDA | ✅ Supported |
| CPU | ✅ Fallback |

Auto-detection priority: **MPS → CUDA → CPU**

## Makefile Targets

```bash
make help            # Show all targets
make install         # Install dependencies
make train-cartpole  # Stage 1
make train-ddqn      # Stage 2
make train-thermal   # Stage 3
make test            # Run all tests
make tensorboard     # Launch TensorBoard
make compare         # DQN vs PID comparison
make clean           # Remove artefacts
```

## References

1. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
3. Wang, Z. et al. (2016). *Dueling Network Architectures for Deep RL.* ICML.
4. Schaul, T. et al. (2016). *Prioritized Experience Replay.* ICLR.
