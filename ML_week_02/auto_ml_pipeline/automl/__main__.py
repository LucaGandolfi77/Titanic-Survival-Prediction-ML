"""
CLI entry-point for AutoML-Lite.

Usage:
    python -m automl fit   --csv data.csv --target Survived
    python -m automl fit   --csv data.csv --target Survived --config configs/fast.yaml
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import AutoMLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="automl",
        description="AutoML-Lite – automatic ML pipeline for tabular data.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── fit ────────────────────────────────────────────────────────
    fit_p = sub.add_parser("fit", help="Run the full pipeline on a CSV.")
    fit_p.add_argument("--csv", required=True, help="Path to input CSV.")
    fit_p.add_argument("--target", required=True, help="Target column name.")
    fit_p.add_argument("--config", default=None, help="YAML config file.")
    fit_p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    fit_p.add_argument("--output-dir", default="outputs", help="Output directory.")

    # ── predict ────────────────────────────────────────────────────
    pred_p = sub.add_parser("predict", help="Predict on new data (not yet implemented).")
    pred_p.add_argument("--csv", required=True)
    pred_p.add_argument("--model", required=True, help="Path to saved model .pkl.")

    args = parser.parse_args()

    if args.command == "fit":
        pipe = AutoMLPipeline(config_path=args.config, output_dir=args.output_dir)
        pipe.fit(csv_path=args.csv, target_column=args.target, test_size=args.test_size)

    elif args.command == "predict":
        print("Predict mode not yet implemented (load model + transform pipeline).")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
