# ecosim/logging_utils.py
from __future__ import annotations
import csv
import os
import sys
from typing import List, Dict

def log_step(step_data: Dict, verbose: bool = True) -> None:
    """Print a concise, human‑readable line for the step."""
    if not verbose:
        return
    step = step_data.get("step", -1)
    grass = step_data.get("grass", 0.0)
    herb = step_data.get("herbivores", 0)
    carn = step_data.get("carnivores", 0)
    msg = (
        f"Step {step:4d} | Grass: {grass:6.1f} | "
        f"Herbivores: {herb:4d} | Carnivores: {carn:3d}"
    )
    # Optional warnings
    warnings = []
    if herb < 5:
        warnings.append("Herbivore pop critically low")
    if carn > 0 and herb > 0 and carn > 0.2 * herb:  # example density flag
        warnings.append("Carnivore density high")
    if warnings:
        msg += " | " + " | ".join(warnings)
    print(msg)

def write_csv(data: List[Dict], filename: str = "ecosim_log.csv") -> None:
    """Write time‑series data to CSV (one row per logged step)."""
    if not data:
        return
    fieldnames = list(data[0].keys())
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def plot_results(csv_path: str = "ecosim_log.csv") -> None:
    """Load CSV and plot populations over time using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print("Matplotlib or pandas not installed; skipping plot.", file=sys.stderr)
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["grass"], label="Grass (units)")
    plt.plot(df["step"], df["herbivores"], label="Herbivores")
    plt.plot(df["step"], df["carnivores"], label="Carnivores")
    plt.xlabel("Time step")
    plt.ylabel("Population / Biomass")
    plt.title("Ecosystem Dynamics")
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.splitext(csv_path)[0] + "_populations.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")
