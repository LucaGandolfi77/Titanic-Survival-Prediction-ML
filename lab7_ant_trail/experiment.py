"""experiment.py — Full experiment pipeline for the Ant Trail Problem.

Runs: N=10, m ∈ {1, 2}, num_games ∈ {1, 5, 10}, model ∈ {dt, mlp, gp}.
Each trained model is evaluated on 5 fixed test boards.
Results are collected in a Pandas DataFrame and saved to CSV.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd

from ant_rec import collect_games
from ant_train import ant_train
from ant_move import play_game


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

N = 10
M_VALUES = [1, 2]
NUM_GAMES_VALUES = [1, 5, 10]
MODEL_TYPES = ["dt", "mlp", "gp"]
NUM_TEST_BOARDS = 5
TEST_BASE_SEED = 9000       # test boards use different seeds from training
TRAIN_BASE_SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"


# -----------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------

def collect_all_data(n: int = N) -> dict[tuple[int, int], str]:
    """Collect training CSVs for every (m, num_games) configuration.

    Returns:
        Mapping (m, num_games) → CSV file path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    paths: dict[tuple[int, int], str] = {}

    for m in M_VALUES:
        for ng in NUM_GAMES_VALUES:
            fname = os.path.join(DATA_DIR, f"train_m{m}_g{ng}.csv")
            if os.path.exists(fname):
                print(f"  [skip] {fname} already exists")
            else:
                print(f"  [collect] m={m}, games={ng} → {fname}")
                scores = collect_games(n, m, ng, fname,
                                       base_seed=TRAIN_BASE_SEED)
                print(f"    Scores: {scores}")
            paths[(m, ng)] = fname

    return paths


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_all_models(
    data_paths: dict[tuple[int, int], str],
    n: int = N,
) -> dict[tuple[int, int, str], Any]:
    """Train one model per (m, num_games, model_type) configuration.

    Returns:
        Mapping (m, num_games, model_type) → trained model.
    """
    models: dict[tuple[int, int, str], Any] = {}

    for (m, ng), csv_path in data_paths.items():
        for mt in MODEL_TYPES:
            key = (m, ng, mt)
            print(f"\n  [train] m={m}, games={ng}, model={mt}")
            t0 = time.time()

            if mt == "gp":
                from gp_player import train_gp
                model = train_gp(n=n, m=m, pop_size=200, n_gen=50,
                                  n_games=ng, base_seed=TRAIN_BASE_SEED,
                                  verbose=True)
            else:
                model = ant_train(csv_path, model_type=mt, augment=True,
                                  verbose=True)

            elapsed = time.time() - t0
            print(f"    ({elapsed:.1f}s)")
            models[key] = model

    return models


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate_models(
    models: dict[tuple[int, int, str], Any],
    n: int = N,
) -> pd.DataFrame:
    """Evaluate every model on *NUM_TEST_BOARDS* fixed test boards.

    Returns:
        DataFrame with columns [m, num_training_games, model_type,
        test_board_id, score].
    """
    records: list[dict[str, Any]] = []

    for (m, ng, mt), model in models.items():
        for t in range(NUM_TEST_BOARDS):
            seed = TEST_BASE_SEED + t
            score = play_game(n, m, model, mt, seed=seed)
            records.append({
                "m": m,
                "num_training_games": ng,
                "model_type": mt,
                "test_board_id": t,
                "score": score,
            })

    df = pd.DataFrame(records)
    return df


# -----------------------------------------------------------------------
# Full experiment
# -----------------------------------------------------------------------

def run_experiment(n: int = N) -> pd.DataFrame:
    """Run the full experiment pipeline: collect → train → evaluate.

    Returns:
        Results DataFrame.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1 — Data Collection")
    print("=" * 60)
    data_paths = collect_all_data(n)

    print("\n" + "=" * 60)
    print("STEP 2 — Model Training")
    print("=" * 60)
    models = train_all_models(data_paths, n)

    print("\n" + "=" * 60)
    print("STEP 3 — Evaluation")
    print("=" * 60)
    df = evaluate_models(models, n)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # Summary table
    summary = (df.groupby(["m", "num_training_games", "model_type"])["score"]
               .agg(["mean", "std", "min", "max"])
               .round(2))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    return df


# -----------------------------------------------------------------------
# Standalone
# -----------------------------------------------------------------------

if __name__ == "__main__":
    df = run_experiment()
