"""
gp_setup.py — DEAP toolbox configuration for prey and predator GP populations.

Biological analogy:
    This module defines the "genetic alphabet" available to each species.
    Terminal sets correspond to an animal's *sensory modalities* (sight,
    proprioception), while function sets are the *neural primitives* the
    brain can combine to form decision circuits.  ``create_prey_toolbox``
    and ``create_predator_toolbox`` are analogous to specifying a species'
    developmental programme.
"""

from __future__ import annotations

import math
import operator
import random as py_random
from functools import partial
from typing import Any, Callable, Tuple

import numpy as np
from deap import base, creator, gp, tools

from config import SimConfig


# ---------------------------------------------------------------------------
# Shared GP primitives (function set)
# ---------------------------------------------------------------------------

def _protected_div(a: float, b: float) -> float:
    """Division that returns 1.0 when the divisor is near zero.

    Prevents NaN / Inf from poisoning GP evaluation.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        ``a / b`` or ``1.0`` if ``|b| < 1e-6``.
    """
    if abs(b) < 1e-6:
        return 1.0
    return a / b


def _if_then_else(condition: float, val_true: float, val_false: float) -> float:
    """Ternary conditional for GP trees.

    Args:
        condition: Treated as boolean (> 0.0 → true).
        val_true:  Returned when condition is true.
        val_false: Returned when condition is false.

    Returns:
        ``val_true`` or ``val_false``.
    """
    return val_true if condition > 0.0 else val_false


def _and(a: float, b: float) -> float:
    """Logical AND for continuous signals (> 0 → true).

    Returns:
        1.0 if both positive, else 0.0.
    """
    return 1.0 if (a > 0.0 and b > 0.0) else 0.0


def _or(a: float, b: float) -> float:
    """Logical OR for continuous signals.

    Returns:
        1.0 if either positive, else 0.0.
    """
    return 1.0 if (a > 0.0 or b > 0.0) else 0.0


def _not(a: float) -> float:
    """Logical NOT for continuous signals.

    Returns:
        0.0 if a > 0, else 1.0.
    """
    return 0.0 if a > 0.0 else 1.0


def _max2(a: float, b: float) -> float:
    """Element-wise max."""
    return a if a >= b else b


def _min2(a: float, b: float) -> float:
    """Element-wise min."""
    return a if a <= b else b


def _safe_mul(a: float, b: float) -> float:
    """Multiplication with overflow guard."""
    r = a * b
    if abs(r) > 1e15:
        return 1e15 if r > 0 else -1e15
    return r


def _safe_add(a: float, b: float) -> float:
    """Addition with overflow guard."""
    r = a + b
    if abs(r) > 1e15:
        return 1e15 if r > 0 else -1e15
    return r


def _safe_sub(a: float, b: float) -> float:
    """Subtraction with overflow guard."""
    r = a - b
    if abs(r) > 1e15:
        return 1e15 if r > 0 else -1e15
    return r


def _erc_random() -> float:
    """Ephemeral random constant in [-1, 1]."""
    return round(py_random.uniform(-1, 1), 3)


def _add_shared_primitives(pset: gp.PrimitiveSet) -> None:
    """Register function-set primitives shared by both species.

    Args:
        pset: DEAP PrimitiveSet to augment.
    """
    pset.addPrimitive(_if_then_else, 3, name="if_then_else")
    pset.addPrimitive(_and, 2, name="and_")
    pset.addPrimitive(_or, 2, name="or_")
    pset.addPrimitive(_not, 1, name="not_")
    pset.addPrimitive(_safe_add, 2, name="add")
    pset.addPrimitive(_safe_sub, 2, name="sub")
    pset.addPrimitive(_safe_mul, 2, name="mul")
    pset.addPrimitive(_protected_div, 2, name="protected_div")
    pset.addPrimitive(_max2, 2, name="max2")
    pset.addPrimitive(_min2, 2, name="min2")


# ---------------------------------------------------------------------------
# Fitness classes (created once globally)
# ---------------------------------------------------------------------------

# Guard: only create once even if module is re-imported.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "PreyIndividual"):
    creator.create("PreyIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
if not hasattr(creator, "PredatorIndividual"):
    creator.create("PredatorIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Prey toolbox
# ---------------------------------------------------------------------------

_prey_pset: gp.PrimitiveSet | None = None


def get_prey_pset(config: SimConfig) -> gp.PrimitiveSet:
    """Return (and cache) the prey PrimitiveSet.

    Terminal set (13 inputs):
        FOOD_N, FOOD_S, FOOD_E, FOOD_W,
        PRED_N, PRED_S, PRED_E, PRED_W,
        PREY_N, PREY_S, PREY_E, PREY_W,
        ENERGY_LOW

    Plus an ephemeral random constant in [-1, 1].

    Args:
        config: Simulation config (unused currently, reserved for future).

    Returns:
        A ``gp.PrimitiveSet``.
    """
    global _prey_pset
    if _prey_pset is not None:
        return _prey_pset

    pset = gp.PrimitiveSet("PREY_BRAIN", 13)
    _add_shared_primitives(pset)

    # Rename arguments to meaningful terminal names.
    pset.renameArguments(
        ARG0="FOOD_N", ARG1="FOOD_S", ARG2="FOOD_E", ARG3="FOOD_W",
        ARG4="PRED_N", ARG5="PRED_S", ARG6="PRED_E", ARG7="PRED_W",
        ARG8="PREY_N", ARG9="PREY_S", ARG10="PREY_E", ARG11="PREY_W",
        ARG12="ENERGY_LOW",
    )

    # Ephemeral random constant.
    pset.addEphemeralConstant("ERC_prey", _erc_random)

    _prey_pset = pset
    return pset


def create_prey_toolbox(config: SimConfig) -> tools.Toolbox:
    """Build a DEAP Toolbox wired for the prey GP population.

    Args:
        config: Simulation config.

    Returns:
        A configured ``Toolbox``.
    """
    pset = get_prey_pset(config)
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=1, max_=config.max_tree_depth)
    toolbox.register("individual", tools.initIterate,
                     creator.PreyIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament,
                     tournsize=config.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0,
                     max_=config.max_tree_depth)
    toolbox.register("mutate", gp.mutUniform,
                     expr=toolbox.expr_mut, pset=pset)

    # Bloat control.
    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth + 2))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth + 2))

    return toolbox


# ---------------------------------------------------------------------------
# Predator toolbox
# ---------------------------------------------------------------------------

_pred_pset: gp.PrimitiveSet | None = None


def get_predator_pset(config: SimConfig) -> gp.PrimitiveSet:
    """Return (and cache) the predator PrimitiveSet.

    Terminal set (10 inputs):
        PREY_N, PREY_S, PREY_E, PREY_W,
        PRED_N, PRED_S, PRED_E, PRED_W,
        CLOSEST_PREY_DIR,
        ENERGY_LOW

    Plus an ephemeral random constant in [-1, 1].

    Args:
        config: Simulation config.

    Returns:
        A ``gp.PrimitiveSet``.
    """
    global _pred_pset
    if _pred_pset is not None:
        return _pred_pset

    pset = gp.PrimitiveSet("PRED_BRAIN", 10)
    _add_shared_primitives(pset)

    pset.renameArguments(
        ARG0="PREY_N", ARG1="PREY_S", ARG2="PREY_E", ARG3="PREY_W",
        ARG4="PRED_N", ARG5="PRED_S", ARG6="PRED_E", ARG7="PRED_W",
        ARG8="CLOSEST_PREY_DIR",
        ARG9="ENERGY_LOW",
    )

    pset.addEphemeralConstant("ERC_pred", _erc_random)

    _pred_pset = pset
    return pset


def create_predator_toolbox(config: SimConfig) -> tools.Toolbox:
    """Build a DEAP Toolbox wired for the predator GP population.

    Args:
        config: Simulation config.

    Returns:
        A configured ``Toolbox``.
    """
    pset = get_predator_pset(config)
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=1, max_=config.max_tree_depth)
    toolbox.register("individual", tools.initIterate,
                     creator.PredatorIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament,
                     tournsize=config.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0,
                     max_=config.max_tree_depth)
    toolbox.register("mutate", gp.mutUniform,
                     expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth + 2))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth + 2))

    return toolbox
