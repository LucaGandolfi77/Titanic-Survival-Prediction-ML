"""
Tree Factory
============
Build sklearn DecisionTreeClassifier instances for every pruning/depth
configuration used in the thesis. Provides a single entry point
``build_tree`` that accepts a strategy name and hyperparameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.tree import DecisionTreeClassifier


def build_tree(
    strategy: str = "none",
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    ccp_alpha: float = 0.0,
    random_state: int = 42,
    **extra: Any,
) -> DecisionTreeClassifier:
    """Create a DecisionTreeClassifier for the given pruning strategy.

    Parameters
    ----------
    strategy : one of "none", "pre_depth", "pre_samples", "ccp", "combined"
    max_depth : tree depth limit (None = unlimited)
    min_samples_leaf : minimum samples at leaf nodes
    min_samples_split : minimum samples to attempt a split
    ccp_alpha : cost-complexity pruning parameter
    random_state : random seed for tie-breaking
    """
    params: Dict[str, Any] = dict(
        criterion="gini",
        random_state=random_state,
    )

    if strategy == "none":
        params.update(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0,
        )
    elif strategy == "pre_depth":
        params.update(
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0,
        )
    elif strategy == "pre_samples":
        params.update(
            max_depth=None,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=0.0,
        )
    elif strategy == "ccp":
        params.update(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=ccp_alpha,
        )
    elif strategy == "combined":
        params.update(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return DecisionTreeClassifier(**params)


def strategy_label(strategy: str) -> str:
    """Human-readable label for plots."""
    labels = {
        "none": "No Pruning",
        "pre_depth": "Pre-Pruning (depth)",
        "pre_samples": "Pre-Pruning (samples)",
        "ccp": "Cost-Complexity",
        "combined": "Combined",
    }
    return labels.get(strategy, strategy)


if __name__ == "__main__":
    tree = build_tree("ccp", ccp_alpha=0.01)
    print(tree.get_params())
