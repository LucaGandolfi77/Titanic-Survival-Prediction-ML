"""
Example: Multiclass classification on the Iris dataset.

Usage:
    python examples/multiclass.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import load_iris

from automl.pipeline import AutoMLPipeline


def main():
    # Load iris and save as CSV
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.replace(" ", "_").replace("_(cm)", "") for c in df.columns]
    csv_path = Path("data/sample/iris.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    pipe = AutoMLPipeline(config_path="configs/fast.yaml", output_dir="outputs/iris")
    pipe.fit(csv_path=str(csv_path), target_column="target")

    print(f"\nMetrics: {pipe.results_['metrics']}")


if __name__ == "__main__":
    main()
