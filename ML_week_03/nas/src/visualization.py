"""Visualisation helpers for the NAS evolutionary search.

Produces:
1. **Fitness over generations** — line plot of best / mean / worst fitness.
2. **Architecture evolution tree** — directed graph showing parent → child
   lineage coloured by fitness.
3. **Population diversity** — depth / param-count distributions per gen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from src.genome import Genome


# ── 1. Fitness over generations ──────────────────────────────────────────────

def plot_fitness_curve(
    history: List[Dict[str, Any]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot best / mean / worst fitness per generation.

    Parameters
    ----------
    history : list[dict]
        Each entry must have ``generation``, ``best``, ``mean``, ``worst``
        keys (floats).
    save_path : str | Path, optional
        If given, save figure to disk; otherwise ``plt.show()``.
    """
    gens = [h["generation"] for h in history]
    best = [h["best"] for h in history]
    mean = [h["mean"] for h in history]
    worst = [h["worst"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, "g-o", label="Best", markersize=4)
    ax.plot(gens, mean, "b--s", label="Mean", markersize=3)
    ax.plot(gens, worst, "r:^", label="Worst", markersize=3)
    ax.fill_between(gens, worst, best, alpha=0.1, color="blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("NAS — Fitness Over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 2. Evolution tree ────────────────────────────────────────────────────────

def plot_evolution_tree(
    all_genomes: List[Genome],
    save_path: Optional[str | Path] = None,
) -> None:
    """Draw a directed graph of parent → child relationships.

    Node colour is mapped to fitness (green = high, red = low).
    Requires ``networkx`` + ``matplotlib``.
    """
    if not _HAS_NX:
        print("networkx not installed — skipping evolution tree.")
        return

    G = nx.DiGraph()
    fitness_map: Dict[str, float] = {}

    for g in all_genomes:
        G.add_node(g.id, generation=g.generation, fitness=g.fitness or 0.0)
        fitness_map[g.id] = g.fitness or 0.0
        if g.parent_ids:
            for pid in g.parent_ids:
                if pid in fitness_map or any(gg.id == pid for gg in all_genomes):
                    G.add_edge(pid, g.id)

    # Layout: group by generation
    pos = {}
    gen_groups: Dict[int, List[str]] = {}
    for node in G.nodes:
        gen = G.nodes[node].get("generation", 0)
        gen_groups.setdefault(gen, []).append(node)
    for gen, nodes in gen_groups.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (gen, -i)

    # Colour by fitness
    fitnesses = [fitness_map.get(n, 0.0) for n in G.nodes]
    vmin = min(fitnesses) if fitnesses else 0
    vmax = max(fitnesses) if fitnesses else 1

    fig, ax = plt.subplots(figsize=(max(14, len(gen_groups) * 2), 8))
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=120,
        node_color=fitnesses,
        cmap=plt.cm.RdYlGn,
        vmin=vmin,
        vmax=vmax,
        with_labels=False,
        arrows=True,
        edge_color="#999999",
        width=0.5,
        alpha=0.9,
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Fitness (val accuracy)")
    ax.set_title("NAS — Architecture Evolution Tree")
    ax.set_xlabel("Generation")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 3. Population diversity ──────────────────────────────────────────────────

def plot_diversity(
    generation_snapshots: Dict[int, List[Dict[str, Any]]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Histogram of depth and param count per generation.

    Parameters
    ----------
    generation_snapshots : dict[int, list[dict]]
        ``gen_num → [{"depth": int, "params": int, "fitness": float}, …]``.
    """
    gens = sorted(generation_snapshots.keys())
    if not gens:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Depth over time
    depths = [[s["depth"] for s in generation_snapshots[g]] for g in gens]
    bp1 = axes[0].boxplot(depths, positions=gens, widths=0.6, patch_artist=True)
    for box in bp1["boxes"]:
        box.set_facecolor("#7CB9E8")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Depth (# layers)")
    axes[0].set_title("Architecture Depth Distribution")
    axes[0].grid(True, alpha=0.3)

    # Param count over time
    params = [[s["params"] for s in generation_snapshots[g]] for g in gens]
    bp2 = axes[1].boxplot(params, positions=gens, widths=0.6, patch_artist=True)
    for box in bp2["boxes"]:
        box.set_facecolor("#FDDA0D")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Trainable Parameters")
    axes[1].set_title("Model Size Distribution")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 4. Interactive Plotly fitness curve (optional) ───────────────────────────

def plotly_fitness_curve(
    history: List[Dict[str, Any]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Interactive Plotly version of the fitness curve."""
    if not _HAS_PLOTLY:
        print("plotly not installed — skipping interactive plot.")
        return

    gens = [h["generation"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=[h["best"] for h in history], name="Best", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=gens, y=[h["mean"] for h in history], name="Mean", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=gens, y=[h["worst"] for h in history], name="Worst", mode="lines+markers"))
    fig.update_layout(
        title="NAS — Fitness Over Generations",
        xaxis_title="Generation",
        yaxis_title="Validation Accuracy",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(str(save_path))
    else:
        fig.show()
