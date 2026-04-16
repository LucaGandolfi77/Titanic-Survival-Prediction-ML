"""SSL strategies sub-package."""

from ssl_strategies.supervised_baseline import supervised_baseline
from ssl_strategies.fully_supervised import fully_supervised
from ssl_strategies.pseudo_labeling import pseudo_labeling
from ssl_strategies.cluster_as_features import cluster_as_features
from ssl_strategies.cluster_prototypes import cluster_prototypes
from ssl_strategies.active_learning import active_learning
from ssl_strategies.label_propagation import label_propagation_strategy
from ssl_strategies.strategy_evaluator import evaluate_strategy

__all__ = [
    "supervised_baseline",
    "fully_supervised",
    "pseudo_labeling",
    "cluster_as_features",
    "cluster_prototypes",
    "active_learning",
    "label_propagation_strategy",
    "evaluate_strategy",
]
