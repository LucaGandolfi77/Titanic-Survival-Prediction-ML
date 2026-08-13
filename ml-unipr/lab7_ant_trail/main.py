"""main.py — Entry point for the Ant Trail Problem project.

Usage examples:
    # 1) Collect training data
    python main.py collect -m 1 --games 5

    # 2) Train a model
    python main.py train --file data/train_m1_g5.csv --model dt

    # 3) Play a game with a trained model (not pickled, quick demo)
    python main.py play -m 1 --model dt --file data/train_m1_g5.csv

    # 4) Run the full experiment pipeline
    python main.py experiment

    # 5) Generate plots from saved results
    python main.py plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def cmd_collect(args: argparse.Namespace) -> None:
    """Collect training games and save to CSV."""
    from ant_rec import collect_games

    os.makedirs(os.path.dirname(args.file) or ".", exist_ok=True)
    scores = collect_games(
        args.n, args.m, args.games, args.file,
        base_seed=args.seed, interactive=args.interactive,
    )
    print(f"Collected {args.games} game(s) → {args.file}")
    print(f"Scores: {scores}  |  Mean: {np.mean(scores):.2f}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train a model on recorded data."""
    from ant_train import ant_train, save_model

    model = ant_train(args.file, model_type=args.model,
                      augment=not args.no_augment, verbose=True)

    if args.save:
        save_model(model, args.save)


def cmd_play(args: argparse.Namespace) -> None:
    """Train a model then play one demo game and report score."""
    from ant_train import ant_train
    from ant_move import play_game

    model = ant_train(args.file, model_type=args.model,
                      augment=not args.no_augment, verbose=True)
    score = play_game(args.n, args.m, model, args.model, seed=args.seed)
    print(f"\n  Demo game score: {score}")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run the full experiment pipeline."""
    from experiment import run_experiment
    from visualize import generate_all_plots

    df = run_experiment(n=args.n)
    generate_all_plots(df)


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots from saved results."""
    from visualize import generate_all_plots
    generate_all_plots()


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ant Trail Problem — ML Lab Assignment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- collect --
    p_col = sub.add_parser("collect", help="Collect training data")
    p_col.add_argument("-n", type=int, default=10)
    p_col.add_argument("-m", type=int, default=1)
    p_col.add_argument("--games", type=int, default=1)
    p_col.add_argument("--file", type=str, default="data/demo.csv")
    p_col.add_argument("--seed", type=int, default=42)
    p_col.add_argument("--interactive", action="store_true")
    p_col.set_defaults(func=cmd_collect)

    # -- train --
    p_tr = sub.add_parser("train", help="Train a model")
    p_tr.add_argument("--file", type=str, required=True)
    p_tr.add_argument("--model", type=str, default="dt",
                      choices=["dt", "mlp", "gp"])
    p_tr.add_argument("--no-augment", action="store_true")
    p_tr.add_argument("--save", type=str, default=None,
                      help="Path to save the pickled model")
    p_tr.set_defaults(func=cmd_train)

    # -- play --
    p_pl = sub.add_parser("play", help="Quick demo game with a trained model")
    p_pl.add_argument("-n", type=int, default=10)
    p_pl.add_argument("-m", type=int, default=1)
    p_pl.add_argument("--file", type=str, required=True)
    p_pl.add_argument("--model", type=str, default="dt",
                      choices=["dt", "mlp", "gp"])
    p_pl.add_argument("--no-augment", action="store_true")
    p_pl.add_argument("--seed", type=int, default=42)
    p_pl.set_defaults(func=cmd_play)

    # -- experiment --
    p_ex = sub.add_parser("experiment", help="Full experiment pipeline")
    p_ex.add_argument("-n", type=int, default=10)
    p_ex.set_defaults(func=cmd_experiment)

    # -- plot --
    p_plt = sub.add_parser("plot", help="Generate plots from results CSV")
    p_plt.set_defaults(func=cmd_plot)

    return parser


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
