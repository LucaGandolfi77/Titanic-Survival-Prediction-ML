"""
predictor.py — Inference Wrapper
=================================
Encapsulates model loading + preprocessing so that the API layer
stays thin and free of ML logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessor, engineer_features, load_pickle

logger = logging.getLogger("titanic_mlops.predictor")


class TitanicPredictor:
    """
    Stateful inference wrapper.

    Loads the preprocessor and trained model once, then exposes a
    ``predict()`` method that accepts raw passenger dictionaries.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        preprocessor_path: Path | None = None,
        model: Any = None,
        preprocessor: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        model_path : Path, optional
            Path to a joblib-serialised sklearn model.
        preprocessor_path : Path, optional
            Path to a pickled ColumnTransformer.
        model : estimator, optional
            Pre-loaded model (takes precedence over *model_path*).
        preprocessor : ColumnTransformer, optional
            Pre-loaded preprocessor (takes precedence over *preprocessor_path*).
        """
        self.model = model
        self.preprocessor = preprocessor

        if self.model is None and model_path is not None:
            logger.info("Loading model from %s", model_path)
            self.model = joblib.load(model_path)

        if self.preprocessor is None and preprocessor_path is not None:
            logger.info("Loading preprocessor from %s", preprocessor_path)
            self.preprocessor = load_pickle(preprocessor_path)

        self._ready = self.model is not None and self.preprocessor is not None
        if self._ready:
            logger.info("Predictor ready")

    # ── Properties ──────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Public API ──────────────────────────────────────────

    def predict(
        self, passengers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run inference on one or more raw passenger records.

        Parameters
        ----------
        passengers : list of dict
            Each dict must contain the keys expected by the preprocessor.

        Returns
        -------
        list of dict
            ``[{"survived": int, "probability": float}, ...]``
        """
        if not self._ready:
            raise RuntimeError("Predictor is not initialised — model or preprocessor missing")

        df = pd.DataFrame(passengers)
        df = engineer_features(df)

        X = self.preprocessor.transform(df)
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)[:, 1]

        results = [
            {"survived": int(p), "probability": round(float(prob), 4)}
            for p, prob in zip(preds, probas)
        ]
        logger.debug("Predicted %d passengers", len(results))
        return results

    def predict_single(self, passenger: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: predict for a single passenger."""
        return self.predict([passenger])[0]
