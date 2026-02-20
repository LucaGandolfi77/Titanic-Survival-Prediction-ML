"""
Quick model screening (Stage 3).

For each candidate model:
  1. Train with default hyper-parameters.
  2. 3-fold CV  →  mean score.
  3. Rank and keep the top-K models for Bayesian HPO.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ..utils.logger import get_logger
from .registry import get_model_entry, list_available_models

logger = get_logger(__name__)


class ModelScreener:
    """Screen many models quickly and pick top-K candidates."""

    def __init__(self, config) -> None:
        self.top_k: int = getattr(config, "top_k", 3)
        self.cv_folds: int = getattr(config, "cv_folds", 3)
        self.metric: str = getattr(config, "metric", "accuracy")
        self.candidates: list[str] = getattr(config, "candidates", list_available_models())
        self.results_: List[dict] = []

    def screen(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        task: str,
    ) -> List[str]:
        """Return top-K model names sorted by descending CV score."""
        scoring = self._resolve_scoring(task)
        results = []

        for name in self.candidates:
            try:
                entry = get_model_entry(name)
                model = entry.build(task)
                t0 = time.perf_counter()
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )
                elapsed = time.perf_counter() - t0
                result = {
                    "model": name,
                    "mean_score": float(scores.mean()),
                    "std_score": float(scores.std()),
                    "time_s": round(elapsed, 2),
                }
                results.append(result)
                logger.info(
                    f"  {name:<25} score={result['mean_score']:.4f} "
                    f"± {result['std_score']:.4f}  ({elapsed:.1f}s)"
                )
            except Exception as exc:
                logger.warning(f"  {name} failed screening: {exc}")

        results.sort(key=lambda r: r["mean_score"], reverse=True)
        self.results_ = results

        top_names = [r["model"] for r in results[: self.top_k]]
        logger.info(f"Top-{self.top_k} models: {top_names}")
        return top_names

    @staticmethod
    def _resolve_scoring(task: str) -> str:
        if task == "classification":
            return "accuracy"
        return "neg_mean_squared_error"

    @property
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results_)
