# ecosim/analysis_example.py
from __future__ import annotations
import statistics
from .config import SimulationConfig
from .simulation import Simulation

def run_multiple_presets() -> None:
    presets = ["balanced", "high_grass", "predator_heavy", "low_regrowth"]
    results = {}
    for preset in presets:
        cfg = SimulationConfig.preset(preset)
        cfg.num_steps = 400  # shorter for example
        cfg.random_seed = 12345  # same seed for fair comparison
        sim = Simulation(cfg)
        data = sim.run()
        final_herb = data[-1]["herbivores"] if data else 0
        final_carn = data[-1]["carnivores"] if data else 0
        results[preset] = {
            "final_herbivores": final_herb,
            "final_carnivores": final_carn,
            "avg_herbivores": statistics.mean(r["herbivores"] for r in data),
            "avg_carnivores": statistics.mean(r["carnivores"] for r in data),
        }
    # Pretty print
    print("Preset comparison (final populations):")
    for preset, vals in results.items():
        print(
            f"{preset:12} | Herbivores: {vals['final_herbivores']:4d} "
            f"| Carnivores: {vals['final_carnivores']:3d} "
            f"| Avg Herb: {vals['avg_herbivores']:5.1f} "
            f"| Avg Carn: {vals['avg_carnivores']:4.1f}"
        )

if __name__ == "__main__":
    run_multiple_presets()
