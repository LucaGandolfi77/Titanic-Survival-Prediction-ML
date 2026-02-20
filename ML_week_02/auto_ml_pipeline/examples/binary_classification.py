"""
Example: Binary classification on Titanic dataset.

Usage:
    python examples/binary_classification.py
"""

from pathlib import Path
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl.pipeline import AutoMLPipeline


def main():
    # Use the Titanic training CSV from the week-01 project
    data_dir = Path(__file__).resolve().parent.parent.parent / "ML_week_01" / "titanic"
    csv_path = data_dir / "train.csv"

    if not csv_path.exists():
        # Fallback: download a small copy
        print(f"Titanic CSV not found at {csv_path}")
        print("Place train.csv in data/sample/ or update the path.")
        return

    pipe = AutoMLPipeline(config_path="configs/fast.yaml", output_dir="outputs/titanic")
    pipe.fit(csv_path=str(csv_path), target_column="Survived")

    print(f"\nMetrics: {pipe.results_['metrics']}")


if __name__ == "__main__":
    main()
