"""PNAS-style predictor network.

Instead of fully training every candidate architecture, the predictor
learns to *estimate* validation accuracy from the genome encoding alone.
After a few initial generations of ground-truth evaluations, the predictor
pre-screens candidates so only the top-*k* most promising architectures
are trained for real.

Architecture
------------
* Genome → fixed-length vector encoding (one-hot layer types + normalised
  params) → LSTM → Linear → predicted accuracy ∈ [0, 1].
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from src.genome import Genome


# ── Genome → tensor encoding ────────────────────────────────────────────────

_LAYER_TYPES = ["conv2d", "maxpool", "avgpool", "batchnorm", "dropout", "dense"]
_ACTIVATIONS = ["relu", "elu"]

_LT_TO_IDX = {lt: i for i, lt in enumerate(_LAYER_TYPES)}
_ACT_TO_IDX = {a: i for i, a in enumerate(_ACTIVATIONS)}

# Per-layer feature vector dimension:
# 6 (one-hot type) + 3 (filters/128, kernel/7, 0) + 2 (activation one-hot)
# + 1 (dropout rate) + 1 (dense units/512) = 13
_GENE_DIM = 13
_MAX_LAYERS = 20  # pad / truncate


def encode_gene(gene) -> List[float]:
    """Encode a single :class:`LayerGene` as a float vector."""
    vec = [0.0] * _GENE_DIM
    # One-hot layer type (6)
    idx = _LT_TO_IDX.get(gene.layer_type, 0)
    vec[idx] = 1.0
    # Normalised numeric params
    vec[6] = gene.params.get("filters", 0) / 128.0
    vec[7] = gene.params.get("kernel_size", 0) / 7.0
    vec[8] = gene.params.get("size", 0) / 4.0
    # Activation one-hot (2)
    act = gene.params.get("activation", "")
    act_idx = _ACT_TO_IDX.get(act, -1)
    if act_idx >= 0:
        vec[9 + act_idx] = 1.0
    # Dropout rate
    vec[11] = gene.params.get("rate", 0.0)
    # Dense units
    vec[12] = gene.params.get("units", 0) / 512.0
    return vec


def encode_genome(genome: Genome) -> torch.Tensor:
    """Encode a full genome as ``(max_layers, gene_dim)`` tensor."""
    rows = [encode_gene(g) for g in genome.layers[:_MAX_LAYERS]]
    # Pad to _MAX_LAYERS
    while len(rows) < _MAX_LAYERS:
        rows.append([0.0] * _GENE_DIM)
    return torch.tensor(rows, dtype=torch.float32)


# ── Predictor model ─────────────────────────────────────────────────────────

class ArchPredictor(nn.Module):
    """LSTM-based predictor: genome encoding → predicted accuracy."""

    def __init__(self, gene_dim: int = _GENE_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(gene_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape: ``(batch, max_layers, gene_dim)``."""
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden)
        return self.head(h_n.squeeze(0)).squeeze(-1)  # (batch,)


# ── Training the predictor ──────────────────────────────────────────────────

class PredictorTrainer:
    """Manages the predictor's lifecycle across NAS generations.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden dimension.
    lr : float
        Learning rate for the predictor's own optimiser.
    """

    def __init__(self, hidden_size: int = 128, lr: float = 1e-3) -> None:
        self.model = ArchPredictor(hidden=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self._history: List[Tuple[torch.Tensor, float]] = []

    def add_observations(self, genomes: List[Genome]) -> None:
        """Record evaluated genomes for later training."""
        for g in genomes:
            if g.fitness is not None:
                self._history.append((encode_genome(g), g.fitness))

    def fit(self, epochs: int = 50, batch_size: int = 32) -> float:
        """Re-train the predictor on all collected observations.

        Returns the final MSE loss.
        """
        if len(self._history) < 5:
            return float("inf")

        self.model.train()
        xs = torch.stack([h[0] for h in self._history])
        ys = torch.tensor([h[1] for h in self._history], dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(xs, ys)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        last_loss = 0.0
        for _ in range(epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                last_loss = loss.item()

        logger.info(f"Predictor trained on {len(self._history)} samples, loss={last_loss:.6f}")
        return last_loss

    @torch.no_grad()
    def predict(self, genomes: List[Genome]) -> List[float]:
        """Return predicted fitness for each genome."""
        self.model.eval()
        xs = torch.stack([encode_genome(g) for g in genomes])
        return self.model(xs).tolist()

    def rank_and_filter(
        self, genomes: List[Genome], top_k: int
    ) -> List[Genome]:
        """Pre-screen: return the *top_k* most promising candidates."""
        scores = self.predict(genomes)
        paired = sorted(zip(scores, genomes), key=lambda t: t[0], reverse=True)
        selected = [g for _, g in paired[:top_k]]
        logger.info(
            f"Predictor pre-screen: {len(genomes)} → {len(selected)} "
            f"(predicted acc range: {paired[0][0]:.4f} – {paired[-1][0]:.4f})"
        )
        return selected
