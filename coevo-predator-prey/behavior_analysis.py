"""
behavior_analysis.py — Automatic detection of emergent strategies.

Biological analogy:
    Ethologists classify animal behaviour by observing movement patterns over
    time.  This module applies the same principle to the agent trajectories
    recorded during simulation episodes: clustering, divergence from the
    group, stationary behaviour near resources, and coordinated multi-agent
    manoeuvres are all quantified and labelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SimConfig
from fitness import AgentSnapshot, EpisodeResult, StepRecord


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class BehaviorReport:
    """Summary of detected emergent behaviours in one episode."""
    flocking: bool = False
    decoying: bool = False
    hiding: bool = False
    pack_hunting: bool = False
    ambushing: bool = False
    herding: bool = False

    def labels(self) -> List[str]:
        """Return a list of string labels for all detected behaviours.

        Returns:
            e.g. ["flocking", "ambushing"].
        """
        out: List[str] = []
        if self.flocking:
            out.append("flocking")
        if self.decoying:
            out.append("decoying")
        if self.hiding:
            out.append("hiding")
        if self.pack_hunting:
            out.append("pack_hunting")
        if self.ambushing:
            out.append("ambushing")
        if self.herding:
            out.append("herding")
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_trajectories(
    episode: EpisodeResult,
    config: SimConfig,
) -> BehaviorReport:
    """Run all behaviour detectors on a recorded episode.

    Args:
        episode: Full trajectory from ``run_episode``.
        config:  Simulation config (thresholds live here).

    Returns:
        A ``BehaviorReport`` with booleans for each detected strategy.
    """
    report = BehaviorReport()

    if not episode.steps:
        return report

    n = config.grid_size

    report.flocking = _detect_flocking(episode, config, n)
    report.decoying = _detect_decoying(episode, config, n)
    report.hiding = _detect_hiding(episode, config, n)
    report.pack_hunting = _detect_pack_hunting(episode, config, n)
    report.ambushing = _detect_ambushing(episode, config, n)
    report.herding = _detect_herding(episode, config, n)

    return report


def analyze_trajectories_labels(
    episode: EpisodeResult,
    config: SimConfig,
) -> List[str]:
    """Convenience wrapper returning just the label strings.

    Args:
        episode: Recorded episode.
        config:  Config.

    Returns:
        List of behaviour label strings.
    """
    return analyze_trajectories(episode, config).labels()


# ---------------------------------------------------------------------------
# Toroidal distance helper
# ---------------------------------------------------------------------------

def _torus_dist(y1: int, x1: int, y2: int, x2: int, n: int) -> float:
    """Manhattan distance on a toroidal grid.

    Args:
        y1, x1, y2, x2: Coordinates.
        n: Grid side length.

    Returns:
        Minimum Manhattan distance on the torus.
    """
    dy = abs(y1 - y2)
    dx = abs(x1 - x2)
    dy = min(dy, n - dy)
    dx = min(dx, n - dx)
    return float(dy + dx)


# ---------------------------------------------------------------------------
# Prey behaviour detectors
# ---------------------------------------------------------------------------

def _detect_flocking(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect flocking: alive prey cluster with avg pairwise distance < threshold.

    Looks for sustained clustering over at least 20% of the episode.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if flocking is detected.
    """
    cluster_steps = 0
    for record in ep.steps:
        alive = [s for s in record.prey_snapshots if s.alive]
        if len(alive) < 2:
            continue
        total_dist = 0.0
        pairs = 0
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                total_dist += _torus_dist(
                    alive[i].y, alive[i].x,
                    alive[j].y, alive[j].x, n)
                pairs += 1
        avg_dist = total_dist / max(pairs, 1)
        if avg_dist < cfg.flock_distance_threshold:
            cluster_steps += 1

    needed = max(1, int(len(ep.steps) * 0.2))
    return cluster_steps >= needed


def _detect_decoying(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect decoying: one prey moves toward a predator while others flee.

    Scans consecutive step-pairs for divergent distance changes.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if decoying is detected.
    """
    decoy_events = 0

    for t in range(1, len(ep.steps)):
        prev = ep.steps[t - 1]
        curr = ep.steps[t]

        alive_prey_prev = [(i, s) for i, s in enumerate(prev.prey_snapshots) if s.alive]
        alive_pred = [s for s in curr.predator_snapshots if s.alive]

        if len(alive_prey_prev) < 2 or not alive_pred:
            continue

        for pred in alive_pred:
            deltas: List[float] = []
            for idx, prey_prev in alive_prey_prev:
                if idx >= len(curr.prey_snapshots):
                    continue
                prey_curr = curr.prey_snapshots[idx]
                if not prey_curr.alive:
                    continue
                dist_prev = _torus_dist(prey_prev.y, prey_prev.x,
                                        pred.y, pred.x, n)
                dist_curr = _torus_dist(prey_curr.y, prey_curr.x,
                                        pred.y, pred.x, n)
                deltas.append(dist_curr - dist_prev)

            if len(deltas) < 2:
                continue

            # At least one prey moving closer and at least one moving away.
            approaching = sum(1 for d in deltas if d < -cfg.decoy_distance_delta)
            fleeing = sum(1 for d in deltas if d > cfg.decoy_distance_delta)
            if approaching >= 1 and fleeing >= 1:
                decoy_events += 1

    return decoy_events >= 3


def _detect_hiding(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect hiding: prey stationary for several steps when no predator nearby.

    We check if any prey stays at the same position for ``ambush_stationary_steps``
    consecutive ticks while no predator is within observation radius.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if hiding is detected.
    """
    num_prey = len(ep.steps[0].prey_snapshots) if ep.steps else 0
    threshold = cfg.ambush_stationary_steps

    for pidx in range(num_prey):
        streak = 0
        prev_y: Optional[int] = None
        prev_x: Optional[int] = None

        for record in ep.steps:
            snap = record.prey_snapshots[pidx]
            if not snap.alive:
                streak = 0
                continue

            if prev_y == snap.y and prev_x == snap.x:
                # Check no predator within prey observation radius.
                pred_near = any(
                    _torus_dist(snap.y, snap.x, ps.y, ps.x, n) <= cfg.prey_obs_radius
                    for ps in record.predator_snapshots if ps.alive
                )
                if not pred_near:
                    streak += 1
                else:
                    streak = 0
            else:
                streak = 1 if snap.alive else 0

            prev_y, prev_x = snap.y, snap.x

            if streak >= threshold:
                return True

    return False


# ---------------------------------------------------------------------------
# Predator behaviour detectors
# ---------------------------------------------------------------------------

def _detect_pack_hunting(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect pack hunting: multiple predators converge on the same prey.

    Measures whether >=2 predators are simultaneously close to the same
    alive prey with angular diversity above threshold.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if pack hunting is detected.
    """
    pack_events = 0

    for record in ep.steps:
        alive_pred = [(i, s) for i, s in enumerate(record.predator_snapshots) if s.alive]
        alive_prey = [s for s in record.prey_snapshots if s.alive]

        if len(alive_pred) < 2 or not alive_prey:
            continue

        for prey in alive_prey:
            close_preds: List[Tuple[float, float]] = []
            for _, ps in alive_pred:
                dist = _torus_dist(ps.y, ps.x, prey.y, prey.x, n)
                if dist <= cfg.predator_obs_radius + 1:
                    dy = (prey.y - ps.y) % n
                    if dy > n // 2:
                        dy -= n
                    dx = (prey.x - ps.x) % n
                    if dx > n // 2:
                        dx -= n
                    angle = math.degrees(math.atan2(dy, dx))
                    close_preds.append((dist, angle))

            if len(close_preds) >= 2:
                angles = [a for _, a in close_preds]
                spread = max(angles) - min(angles)
                if spread > cfg.pack_angle_diversity_threshold:
                    pack_events += 1

    return pack_events >= 3


def _detect_ambushing(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect ambushing: a predator stays near a food cluster waiting for prey.

    We look for a predator stationary for ``ambush_stationary_steps``
    consecutive ticks near cells with food.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if ambushing is detected.
    """
    num_pred = len(ep.steps[0].predator_snapshots) if ep.steps else 0
    threshold = cfg.ambush_stationary_steps

    for pidx in range(num_pred):
        streak = 0
        prev_y: Optional[int] = None
        prev_x: Optional[int] = None

        for record in ep.steps:
            snap = record.predator_snapshots[pidx]
            if not snap.alive:
                streak = 0
                continue

            stationary = (prev_y == snap.y and prev_x == snap.x)
            # We approximate "near food" by checking if food_count > 0 in episode
            # (we don't store per-cell food in snapshot, so use heuristic).
            near_food = record.food_count > 0

            if stationary and near_food:
                streak += 1
            else:
                streak = 0

            prev_y, prev_x = snap.y, snap.x

            if streak >= threshold:
                return True

    return False


def _detect_herding(ep: EpisodeResult, cfg: SimConfig, n: int) -> bool:
    """Detect herding: predators spread out to cut off prey escape routes.

    Requires average pairwise predator distance above ``herd_spread_threshold``
    while prey are clustered.

    Args:
        ep:  Episode data.
        cfg: Config.
        n:   Grid size.

    Returns:
        True if herding is detected.
    """
    herd_steps = 0

    for record in ep.steps:
        alive_pred = [s for s in record.predator_snapshots if s.alive]
        alive_prey = [s for s in record.prey_snapshots if s.alive]

        if len(alive_pred) < 2 or len(alive_prey) < 2:
            continue

        # Predator spread.
        pred_dist_total = 0.0
        pred_pairs = 0
        for i in range(len(alive_pred)):
            for j in range(i + 1, len(alive_pred)):
                pred_dist_total += _torus_dist(
                    alive_pred[i].y, alive_pred[i].x,
                    alive_pred[j].y, alive_pred[j].x, n)
                pred_pairs += 1
        pred_avg = pred_dist_total / max(pred_pairs, 1)

        # Prey clustering.
        prey_dist_total = 0.0
        prey_pairs = 0
        for i in range(len(alive_prey)):
            for j in range(i + 1, len(alive_prey)):
                prey_dist_total += _torus_dist(
                    alive_prey[i].y, alive_prey[i].x,
                    alive_prey[j].y, alive_prey[j].x, n)
                prey_pairs += 1
        prey_avg = prey_dist_total / max(prey_pairs, 1)

        if pred_avg > cfg.herd_spread_threshold and prey_avg < cfg.flock_distance_threshold * 2:
            herd_steps += 1

    needed = max(1, int(len(ep.steps) * 0.15))
    return herd_steps >= needed
