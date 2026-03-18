# Ant Trail Problem — ML Lab Assignment

An NxN grid-based game where an **ant** collects food cells.
Three types of player models are trained and compared: **Decision Tree**, **MLP**, and **Genetic Programming** (DEAP).

## Project Structure

| File | Description |
|---|---|
| `board.py` | NxN grid board: food placement, querying, display |
| `ant.py` | Ant agent: movement, scoring, neighbourhood observation |
| `ant_rec.py` | Record games (interactive via curses or automated heuristic) |
| `ant_train.py` | Train supervised models (DT, MLP) with symmetry augmentation |
| `ant_move.py` | Use a trained model to play a full game |
| `gp_player.py` | Evolve a GP player with DEAP |
| `experiment.py` | Full pipeline: collect → train → evaluate over all configs |
| `visualize.py` | Grouped bar charts and box plots |
| `main.py` | CLI entry point |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect 5 games with neighbourhood m=1
python main.py collect -m 1 --games 5 --file data/train_m1_g5.csv

# Train a Decision Tree
python main.py train --file data/train_m1_g5.csv --model dt

# Play a demo game
python main.py play -m 1 --model dt --file data/train_m1_g5.csv

# Run the full experiment (all 18 configurations)
python main.py experiment

# Re-generate plots from saved results
python main.py plot
```

## Game Rules

- **Board**: NxN grid, N food cells (+1), empty cells (0).
- **Ant**: starts on a random empty cell (marked -1).
- **Movement**: up (row−1), down (row+1), left (col−1), right (col+1).
- **Scoring**: landing on a cell earns its value; the cell is then decremented.
- **Out-of-bounds**: ant dies, loses (N+2) points.
- **Max moves**: 2N per game.
- **Neighbourhood**: (2m+1)×(2m+1) centred on the ant; OOB cells = −(N+2).

## Models

| Type | Details |
|---|---|
| **DT** | `DecisionTreeClassifier(max_depth=10)` from sklearn |
| **MLP** | `MLPClassifier(hidden_layer_sizes=(64,32), relu, max_iter=500)` |
| **GP** | DEAP: pop=200, gen=50, tournament(5), subtree crossover/mutation, height≤12 |

### GP Output → Direction Mapping

```
output ≤ −K  → up
−K < output ≤ 0  → down
 0 < output ≤ K  → right
output > K  → left       (K = 1.0)
```

### Fallback Rule
If **all** neighbourhood values ≤ 0, pick a random direction excluding the reverse of the last move.

### Data Augmentation
Each (neighbourhood, direction) pair is rotated by 90°/180°/270° → 4× training data.

## Experiment Design

- **N** = 10
- **m** ∈ {1, 2}
- **num_games** ∈ {1, 5, 10}
- **6 data configurations** × **3 model types** = **18 trained models**
- Each model tested on **5 fixed boards** (different from training seeds)
- Metrics reported: mean, std, min, max score
- Output: grouped bar chart + box plot saved in `results/`
