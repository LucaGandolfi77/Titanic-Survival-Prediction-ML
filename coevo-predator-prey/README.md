# Co-evolutionary Predator-Prey Simulation

A Genetic Programming (GP) simulation of competitive co-evolution between
**predator** and **prey** populations on a discrete toroidal grid world, built
with [DEAP](https://deap.readthedocs.io/).

## Research question

> Can the Red Queen dynamic — where each species must continuously adapt
> simply to maintain relative fitness — produce emergent group behaviours
> (flocking, pack hunting, herding, …) from individually evolved decision
> trees?

---

## Architecture

| Module                | Purpose                                               |
|-----------------------|-------------------------------------------------------|
| `config.py`           | Centralised `SimConfig` dataclass and named presets   |
| `world.py`            | Toroidal grid, food placement, directional queries    |
| `agents.py`           | Prey / Predator classes, observation & action decode  |
| `gp_setup.py`         | DEAP primitives, toolboxes, bloat-control decorators  |
| `fitness.py`          | Episode runner and fitness functions for both species  |
| `coevolution.py`      | Co-evolutionary engine with opponent sampling          |
| `behavior_analysis.py`| Automatic detection of 6 emergent strategies          |
| `logging_utils.py`    | CSV logging, console output, GP tree serialisation    |
| `visualization.py`    | Fitness curves, behaviour timeline, GIF animation     |
| `main.py`             | CLI entry point                                       |

---

## Emergent behaviours detected

### Prey strategies

| Behaviour | Description                                           | Detection criterion                              |
|-----------|-------------------------------------------------------|--------------------------------------------------|
| Flocking  | Prey cluster tightly together                         | Avg pairwise distance < `flock_distance_threshold` for ≥20% steps |
| Decoying  | One prey approaches predator while others flee        | Divergent distance changes >= 3 events           |
| Hiding    | Prey stays stationary near food when no predator near | Stationary for `ambush_stationary_steps` ticks    |

### Predator strategies

| Behaviour    | Description                                      | Detection criterion                               |
|--------------|--------------------------------------------------|---------------------------------------------------|
| Pack hunting | Multiple predators converge from different angles | Angle diversity > `pack_angle_diversity_threshold` |
| Ambushing    | Predator waits near food for prey to approach     | Stationary near food for `ambush_stationary_steps` |
| Herding      | Predators spread to cut off prey escape routes    | Pred spread > `herd_spread_threshold` while prey clustered |

---

## Installation

```bash
# Clone / enter the directory
cd coevo-predator-prey

# Create a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default settings
python main.py

# Choose a named preset
python main.py --preset BLIND_PREY

# Customise
python main.py --preset SYMMETRIC --generations 50 --seed 7 --pop-size 100

# Generate episode animation (requires pillow)
python main.py --animate
```

### Available presets

| Preset          | Prey radius | Predator radius |
|-----------------|-------------|-----------------|
| `DEFAULT`       | 3           | 2               |
| `BLIND_PREY`    | 1           | 2               |
| `BLIND_PREDATOR`| 3           | 1               |
| `SYMMETRIC`     | 2           | 2               |
| `ASYMMETRIC`    | 4           | 1               |

### Outputs

All outputs are saved to `output/` (configurable via `--output-dir`):

- `generations.csv` — per-generation fitness, survival, kill statistics
- `fitness_curves.png` — prey vs. predator mean fitness over time
- `behavior_emergence.png` — heatmap of detected behaviours per generation
- `trees/gen_NNNNN.txt` — best GP tree expressions every N generations
- `episode.gif` — animated final episode (with `--animate`)

---

## GP design

- **Shared function set:** `if_then_else`, `and_`, `or_`, `not_`, `add`,
  `sub`, `mul`, `protected_div`, `max2`, `min2`
- **Prey terminals (13):** `FOOD_N, FOOD_S, FOOD_E, FOOD_W`,
  `PRED_N, PRED_S, PRED_E, PRED_W`, `PREY_N, PREY_S, PREY_E, PREY_W`,
  `ENERGY_LOW`
- **Predator terminals (10):** `PREY_N, PREY_S, PREY_E, PREY_W`,
  `PRED_N, PRED_S, PRED_E, PRED_W`, `CLOSEST_PREY_DIR`, `ENERGY_LOW`
- **Action decode:** GP output mapped to 5 directions via threshold bands:
  `NORTH (< -0.6)`, `WEST (-0.6..−0.2)`, `STAY (−0.2..0.2)`,
  `EAST (0.2..0.6)`, `SOUTH (> 0.6)`

---

## The Red Queen hypothesis

In Van Valen's Red Queen model, species in competitive relationships must
continuously evolve to keep pace with each other.  This simulation provides a
computational laboratory to observe this dynamic: improving prey survival
strategies exerts pressure on predators to evolve better hunting, which in
turn pushes prey to adapt further — potentially producing increasingly
sophisticated emergent behaviours that neither population was explicitly
designed to exhibit.

---

## License

MIT
