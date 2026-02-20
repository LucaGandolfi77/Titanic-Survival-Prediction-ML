"""
Example: Regression on the California Housing dataset.

Usage:
    python examples/regression.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import fetch_california_housing

from automl.pipeline import AutoMLPipeline


def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    # Subsample for speed
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    csv_path = Path("data/sample/california_housing.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    pipe = AutoMLPipeline(config_path="configs/fast.yaml", output_dir="outputs/housing")
    pipe.fit(csv_path=str(csv_path), target_column="MedHouseVal")

    print(f"\nMetrics: {pipe.results_['metrics']}")


if __name__ == "__main__":
    main()
