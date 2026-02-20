"""
Text feature extraction via TF-IDF.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextFeatureEngineer:
    """TF-IDF (or count) vectorization for text columns."""

    def __init__(self, config) -> None:
        self.enabled: bool = getattr(config, "enabled", True)
        self.method: str = getattr(config, "method", "tfidf")
        self.max_features: int = getattr(config, "max_features", 100)
        ngram = getattr(config, "ngram_range", [1, 2])
        self.ngram_range = tuple(ngram) if isinstance(ngram, list) else ngram
        self._vectorizers: dict[str, TfidfVectorizer] = {}
        self._input_cols: List[str] = []
        self._output_names: List[str] = []

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self._input_cols = list(X.columns)
        parts: list[np.ndarray] = []
        names: List[str] = []

        for col in self._input_cols:
            corpus = X[col].fillna("").astype(str)
            vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words="english",
            )
            mat = vec.fit_transform(corpus).toarray()
            self._vectorizers[col] = vec
            parts.append(mat)
            names.extend([f"{col}_tfidf_{t}" for t in vec.get_feature_names_out()])

        self._output_names = names
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        for col in self._input_cols:
            corpus = X[col].fillna("").astype(str)
            mat = self._vectorizers[col].transform(corpus).toarray()
            parts.append(mat)
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)

    @property
    def feature_names(self) -> List[str]:
        return self._output_names
