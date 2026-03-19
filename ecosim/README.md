ecosim – Discrete‑Time Grass‑Herbivore‑Carnivore Ecosystem Simulator
A lightweight, extensible, object‑oriented Python project that simulates a simple trophic chain (grass → herbivores → carnivores) in discrete time steps. The simulator is designed for easy experimentation, clear logging, and straightforward extension for research or teaching purposes.

Table of Contents
Overview

Features

Project Structure

Installation

Quick Start

Configuration

Running the Simulation

Logging and Output

Plotting Results

Example Scripts

Extending the Project

Design Notes & Ecological Rationale

Citation

License

Overview
ecosim implements a discrete‑time, agent‑based model where:

Grass is a renewable biomass pool that regrows toward a carrying capacity.

Herbivores are individual agents that consume grass, survive if they meet a minimum intake, starve after a configurable number of low‑intake steps, and reproduce when well‑fed.

Carnivores are individual agents that hunt herbivores, track a sliding window of recent kills, survive only if they meet a kill quota over that window, and reproduce conservatively to avoid predator‑driven prey collapse.

The simulation proceeds step‑by‑step (one step = one day, or any chosen unit) and logs a rich set of metrics for later analysis.

Features
Modular OO design – separate modules for configuration, entities, environment, simulation orchestration, and logging.

Centralized configuration – @dataclass based SimulationConfig with presets (balanced, high_grass, predator_heavy, low_regrowth).

Fine‑grained ecological rules – grass regrowth, herbivore intake & starvation, carnivore kill window, density‑dependent hunting success, conservative predator reproduction.

Deterministic & stochastic modes – controllable random seed for reproducibility.

Rich logging – per‑step console output and CSV file (grass amount, population counts, births/deaths, average intake, average kills, warnings).

Optional plotting – one‑click generation of population trajectory plots via matplotlib (if installed).

Command‑line interface – argparse driven main.py for quick runs and parameter sweeps.

Example analysis script – demonstrates running multiple configurations and comparing outcomes.

Easy to extend – add new species, modify rules, or plug in alternative logging/storage backends.

Project Structure
text
ecosim/
├─ config.py          # Configuration dataclasses and presets
├─ entities.py        # Herbivore and Carnivore agent classes
├─ environment.py     # Ecosystem state and one‑step update logic
├─ simulation.py      # Simulation orchestrator (runs multiple steps)
├─ logging_utils.py   # Console logging, CSV writing, plotting helper
├─ main.py            # Entry point with CLI argument parsing
├─ analysis_example.py# Example: run several presets and compare results
└─ README.py          # Executable README (prints formatted docs)
Installation
Obtain the code

bash
git clone https://github.com/yourusername/ecosim.git   # or copy the folder
cd ecosim
(Optional) Create a virtual environment

bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
Install dependencies
The core simulation runs with only the Python standard library.
For plotting, install the optional packages:

bash
pip install matplotlib pandas
Quick Start
Run a default balanced simulation for 500 steps:

bash
python -m ecosim.main
You will see a concise line printed for each logged step, a file ecosim_log.csv containing the full time‑series, and (if matplotlib is available) a PNG plot ecosim_log_populations.png.

Configuration
All tunable parameters live in ecosim/config.py. The top‑level SimulationConfig groups:

Group	Parameters	Description
Initial populations	initial_grass_amount, initial_herbivores, initial_carnivores	Starting values at step 0
Grass dynamics	grass_carrying_capacity, grass_regrowth_rate, grass_consumption_per_herbivore	Logistic‑like regrowth toward carrying capacity; per‑herbivore max consumption
Herbivore dynamics	herbivore_min_intake_to_survive, herbivore_intake_for_reproduction, herbivore_starvation_steps_before_death, herbivore_base_reproduction_rate	Intake thresholds, starvation memory, reproduction chance
Carnivore dynamics	carnivore_required_kills_per_window, carnivore_kill_window_length, carnivore_reproduction_rate, carnivore_max_density_factor (optional), carnivore_base_hunt_success_prob, herbivore_safety_threshold, critical_herbivore_level	Kill quota, sliding window, low base reproduction, density cap, hunt success modulation
Simulation controls	num_steps, random_seed, logging_frequency	Length of run, RNG seed, how often to log to CSV/console
Presets
Call SimulationConfig.preset("<name>") to get a ready‑made configuration:

balanced – default values (see source).

high_grass – doubled carrying capacity and faster regrowth.

predator_heavy – more initial carnivores and higher predator reproduction.

low_regrowth – slow grass regrowth to test system fragility.

You can also create a custom config by instantiating SimulationConfig directly and overriding any field.

Running the Simulation
Via Command Line
bash
python -m ecosim.main [options]
Options

Option	Argument	Description
--num-steps	INT	Override number of time steps
--random-seed	INT	Seed for reproducibility
--preset	{balanced,high_grass,predator_heavy,low_regrowth}	Configuration preset (default: balanced)
--log-file	PATH	CSV file for detailed logs (default: ecosim_log.csv)
--no-plot	FLAG	Skip generation of population plot
--quiet	FLAG	Suppress per‑step console output
-h, --help	—	Show help message
Examples

bash
# Run 1000 steps with the predator_heavy preset, seed 42
python -m ecosim.main --num-steps 1000 --preset predator_heavy --random-seed 42

# Quiet run, log to custom file, no plot
python -m ecosim.main --quiet --log-file run_a.csv --no-plot
As a Library
python
from ecosim.config import SimulationConfig
from ecosim.simulation import Simulation

cfg = SimulationConfig.preset("high_grass")
cfg.num_steps = 800
cfg.random_seed = 2026

sim = Simulation(cfg)
results = sim.run()   # list of dicts, one per logged step
# results can be fed to pandas, plotted, etc.
Logging and Output
At every step (or every logging_frequency steps) the simulator records:

Metric	Meaning
step	Time step index
grass	Total grass biomass (units)
grass_fraction	Fraction of carrying capacity
herbivores	Number of herbivore agents
carnivores	Number of carnivore agents
herb_births / herb_deaths	Births and deaths this step
carn_births / carn_deaths	Births and deaths this step
avg_herb_intake	Mean grass eaten per herbivore last step
avg_carn_kills	Mean kills per carnivore over its kill window
total_killed_this_step	Total herbivores killed by carnivores this step
Warning flags (added to console line)	e.g., low herbivore pop, high carnivore density
Console
A short, human‑readable line is printed (unless --quiet):

text
Step  0 | Grass: 500.0 | Herbivores: 50 | Carnivores: 10
Step  1 | Grass: 495.2 | Herbivores: 48 | Carnivores: 10 | Herbivore pop critically low
...
CSV
logging_utils.write_csv() writes one row per logged step to the file given by --log-file. The file can be loaded directly with pandas.read_csv() or the built‑in csv module for downstream analysis.

Plotting Results
If matplotlib and pandas are installed, call:

bash
python -m ecosim.main   # creates plot automatically unless --no-plot
or, after a run:

python
from ecosim.logging_utils import plot_results
plot_results("ecosim_log.csv")
The function generates a PNG named <csv_base>_populations.png showing three lines:

Grass (units) – biomass over time

Herbivores – population count

Carnivores – population count

You can easily adapt the plotting function to add moving averages, death breakdowns, etc.

Example Scripts
analysis_example.py
Demonstrates how to run multiple presets with the same seed and compare summary statistics:

bash
python -m ecosim.analysis_example
Sample output:

text
Preset comparison (final populations):
balanced      | Herbivores:  42 | Carnivores:  7 | Avg Herb: 45.3 | Avg Carn: 6.8
high_grass    | Herbivores:  68 | Carnivores:  9 | Avg Herb: 71.2 | Avg Carn: 8.5
predator_heavy| Herbivores:  30 | Carnivores: 12 | Avg Herb: 32.1 | Avg Carn:11.0
low_regrowth  | Herbivores:  15 | Carnivores:  2 | Avg Herb: 16.4 | Avg Carn: 1.9
Feel free to modify this script to sweep any parameter (e.g., grass regrowth rate) and store results for statistical analysis.

Extending the Project
Because the code is modular, you can extend it in many ways:

New Species / Resources

Subclass Herbivore or Carnivore in entities.py to add traits (e.g., omnivores, scavengers).

Add a new resource class (e.g., Water) and update environment.py to include its dynamics.

Alternative Rules

Replace the grass regrowth in environment._regrow_grass() with a true logistic function: r * G * (1 - G/K).

Modify hunting success to depend on a more complex functional response (e.g., Holling type II).

Implement age‑dependent reproduction or senescence.

Different Logging / Storage

Edit logging_utils.py to write JSON, SQLite, or push to a monitoring system (Prometheus, InfluxDB).

Add custom metrics (e.g., variance of intake, Gini coefficient) by extending the stats dictionary in environment.py.

Parallel Experiments

Use Python’s multiprocessing or joblib to run many simulations with different configurations and aggregate results.

The analysis_example.py file provides a simple template for such parameter sweeps.

Visualization Enhancements

Integrate plotly for interactive plots, or generate animated GIFs with matplotlib.animation.

Create phase‑plane plots (herbivores vs. carnivores) or scatter plots of intake vs. reproduction.

All extension points are clearly marked with comments in the source code.

Design Notes & Ecological Rationale
Grass Regrowth – A simple linear‑fractional approach toward carrying capacity ensures biomass never exceeds realistic limits while remaining computationally cheap.

Herbivore Starvation Counter – Captures the idea that organisms can tolerate short periods of low food but perish after prolonged shortage.

Reproduction Probability – Increases with intake above a threshold, reflecting the energetic cost of reproduction.

Carnivore Kill Window – Mimics the need for predators to obtain sufficient prey over a recent period to survive; the sliding window makes survival dependent on recent hunting success rather than a single step.

Conservative Predator Reproduction – Prevents runaway predator growth that would drive herbivores to extinction, a common pitfall in naive predator‑prey models. The implementation additionally reduces reproduction when kill rates are excessively high and increases mortality when herbivores become scarce.

Density‑Dependent Hunting Success – The base hunt probability is scaled by the herbivore‑to‑carnivore ratio, giving higher success when prey are abundant and lower success when predators are numerous, which promotes emergent oscillations.

These choices are documented with inline comments and aim to produce interesting dynamics (stable equilibria, limit cycles, or transient chaos) that are useful for educational demonstrations or as a testbed for more complex ecological modeling.

Citation
The overall design of logging, modularity, and discrete‑time updates follows common practices in agent‑based ecosystem simulators .
​

Ecosim Toolbox: Animal Ecosystem Simulation documentation
​
https://ecosim.readthedocs.io

License
This project is released under the MIT License – see the LICENSE file for details.

Happy simulating! 🌱🦌🐺