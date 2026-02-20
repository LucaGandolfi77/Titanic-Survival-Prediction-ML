"""
AutoMLPipeline – main orchestrator.

6-stage pipeline:
  1. Auto-EDA (profile + impute + outliers)
  2. Feature Engineering (numeric / categorical / datetime / text + selection)
  3. Model Screening (quick CV → top-K)
  4. Bayesian HPO (Optuna per model)
  5. Ensemble (Voting / Stacking)
  6. Output (model .pkl + HTML report + FastAPI code)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .eda.missing_handler import MissingValueHandler
from .eda.outlier_detector import OutlierDetector
from .eda.profiler import DataProfiler
from .eda.type_detector import TypeDetector
from .evaluation.analyzer import ModelAnalyzer
from .evaluation.cross_validator import CrossValidator
from .evaluation.metrics import compute_metrics
from .features.categorical import CategoricalFeatureEngineer
from .features.datetime_features import DateTimeFeatureEngineer
from .features.numeric import NumericFeatureEngineer
from .features.selector import FeatureSelector
from .features.text_features import TextFeatureEngineer
from .models.ensemble import EnsembleBuilder
from .models.optimizer import BayesianOptimizer
from .models.screener import ModelScreener
from .reporting.api_generator import APIGenerator
from .reporting.html_report import HTMLReportGenerator
from .utils.config import load_config
from .utils.logger import get_logger
from .utils.serializer import save_model

logger = get_logger(__name__)


class AutoMLPipeline:
    """End-to-end pipeline: CSV in → trained model + report out."""

    def __init__(self, config_path: Optional[str] = None, **overrides) -> None:
        self.cfg = load_config(config_path)
        # Apply any runtime overrides
        for key, val in overrides.items():
            setattr(self.cfg, key, val)

        self.results_: Dict[str, Any] = {}
        self._fitted = False

        # Sub-components (initialised lazily from config)
        self._type_detector: Optional[TypeDetector] = None
        self._profiler: Optional[DataProfiler] = None
        self._missing_handler: Optional[MissingValueHandler] = None
        self._outlier_detector: Optional[OutlierDetector] = None
        self._numeric_eng: Optional[NumericFeatureEngineer] = None
        self._cat_eng: Optional[CategoricalFeatureEngineer] = None
        self._dt_eng: Optional[DateTimeFeatureEngineer] = None
        self._text_eng: Optional[TextFeatureEngineer] = None
        self._selector: Optional[FeatureSelector] = None
        self._feature_names: List[str] = []

    # ── Public API ────────────────────────────────────────────────

    def fit(
        self,
        csv_path: str,
        target_column: str,
        test_size: float = 0.2,
    ) -> "AutoMLPipeline":
        """Run the full 6-stage pipeline."""
        t_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("AutoML-Lite  — starting pipeline")
        logger.info("=" * 60)

        # ── Load data ─────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {csv_path}  shape={df.shape}")
        self.results_["dataset_shape"] = f"{df.shape[0]} rows × {df.shape[1]} cols"

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Detect task
        task = self._infer_task(y)
        self.results_["task"] = task
        logger.info(f"Task detected: {task}")

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task == "classification" else None,
        )

        # ── Stage 1: Auto-EDA ────────────────────────────────────
        logger.info("\n[Stage 1/6] Auto-EDA")
        self._type_detector = TypeDetector()
        col_types = self._type_detector.detect(X_train)
        logger.info(f"  Column types: { {k: len(v) for k, v in col_types.items()} }")

        self._profiler = DataProfiler(self.cfg.eda)
        profile = self._profiler.profile(X_train)
        self.results_["profile_summary"] = profile.get("summary", {})

        self._missing_handler = MissingValueHandler(self.cfg.eda.missing)
        X_train = self._missing_handler.fit_transform(X_train, col_types)
        X_test = self._missing_handler.transform(X_test)

        self._outlier_detector = OutlierDetector(self.cfg.eda.outliers)
        X_train = self._outlier_detector.fit_transform(X_train, col_types.get("numeric", []))
        X_test = self._outlier_detector.transform(X_test)

        # ── Stage 2: Feature Engineering ──────────────────────────
        logger.info("\n[Stage 2/6] Feature Engineering")
        X_train_arr, X_test_arr, feat_names = self._engineer_features(
            X_train, X_test, y_train, col_types, task,
        )
        self._feature_names = feat_names

        # ── Stage 3: Model Screening ─────────────────────────────
        logger.info("\n[Stage 3/6] Model Screening")
        screener = ModelScreener(self.cfg.screening)
        top_models = screener.screen(X_train_arr, y_train, task)
        self.results_["screening_results"] = screener.results_

        # ── Stage 4: Bayesian HPO ─────────────────────────────────
        logger.info("\n[Stage 4/6] Bayesian Hyper-parameter Optimisation")
        optimizer = BayesianOptimizer(self.cfg.hpo)
        best_models = optimizer.optimize(X_train_arr, y_train, top_models, task)
        self.results_["best_params"] = optimizer.best_params_
        self.results_["best_scores"] = optimizer.best_scores_

        # ── Stage 5: Ensemble ─────────────────────────────────────
        logger.info("\n[Stage 5/6] Ensemble Building")
        ens_cfg = self.cfg.ensemble
        if getattr(ens_cfg, "enabled", True) and len(best_models) > 1:
            builder = EnsembleBuilder(ens_cfg)
            final_model = builder.build(best_models, X_train_arr, y_train, task)
            self.results_["ensemble_method"] = builder.method
        else:
            # Use single best model
            best_name = max(optimizer.best_scores_, key=optimizer.best_scores_.get)
            final_model = best_models[best_name]
            self.results_["ensemble_method"] = f"single ({best_name})"

        self._final_model = final_model

        # ── Stage 6: Evaluate & Output ────────────────────────────
        logger.info("\n[Stage 6/6] Evaluation & Output")
        y_pred = final_model.predict(X_test_arr)
        y_proba = None
        if hasattr(final_model, "predict_proba"):
            try:
                y_proba = final_model.predict_proba(X_test_arr)
            except Exception:
                pass

        metrics = compute_metrics(y_test.values, y_pred, task, y_proba)
        self.results_["metrics"] = metrics

        # Feature importance & charts
        analyzer = ModelAnalyzer(self.cfg.get("analysis", self.cfg))
        imp_df = analyzer.feature_importance(final_model, self._feature_names)
        self.results_["feature_importance"] = imp_df
        self.results_["importance_chart"] = analyzer.importance_chart(imp_df)

        # Save model
        output_dir = Path(getattr(self.cfg, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        save_model(final_model, str(model_path))

        # HTML report
        report_cfg_data = {
            "output_path": str(output_dir / "report.html"),
            "title": f"AutoML-Lite — {Path(csv_path).stem}",
        }

        class _ReportCfg:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        reporter = HTMLReportGenerator(_ReportCfg(report_cfg_data))
        reporter.generate(self.results_)

        # FastAPI code
        api_cfg_data = {"output_path": str(output_dir / "app.py")}
        api_gen = APIGenerator(_ReportCfg(api_cfg_data))
        api_gen.generate(self._feature_names, str(model_path), task)

        elapsed = time.perf_counter() - t_start
        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"  Model  → {model_path}")
        logger.info(f"  Report → {output_dir / 'report.html'}")
        logger.info(f"  API    → {output_dir / 'app.py'}")
        logger.info("=" * 60)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data and predict."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
        # We need to apply the same transforms
        col_types = self._type_detector.detect(X)
        X = self._missing_handler.transform(X)
        X = self._outlier_detector.transform(X)
        X_arr = self._transform_features(X, col_types)
        return self._final_model.predict(X_arr)

    # ── Internal ──────────────────────────────────────────────────

    def _infer_task(self, y: pd.Series) -> str:
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification"
        if y.nunique() <= 20:
            return "classification"
        return "regression"

    def _engineer_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        col_types: dict,
        task: str,
    ):
        """Apply numeric/cat/dt/text transforms, concat, select."""
        parts_train: list = []
        parts_test: list = []
        all_names: List[str] = []

        # Numeric
        num_cols = col_types.get("numeric", [])
        if num_cols:
            # Filter to columns that actually exist
            num_cols = [c for c in num_cols if c in X_train.columns]
            if num_cols:
                self._numeric_eng = NumericFeatureEngineer(self.cfg.features.numeric)
                arr_tr = self._numeric_eng.fit_transform(X_train[num_cols])
                arr_te = self._numeric_eng.transform(X_test[num_cols])
                parts_train.append(arr_tr)
                parts_test.append(arr_te)
                all_names.extend(self._numeric_eng.feature_names)

        # Categorical
        cat_cols = col_types.get("categorical", []) + col_types.get("boolean", [])
        cat_cols = [c for c in cat_cols if c in X_train.columns]
        if cat_cols:
            self._cat_eng = CategoricalFeatureEngineer(self.cfg.features.categorical)
            arr_tr = self._cat_eng.fit_transform(X_train[cat_cols], y_train)
            arr_te = self._cat_eng.transform(X_test[cat_cols])
            parts_train.append(arr_tr)
            parts_test.append(arr_te)
            all_names.extend(self._cat_eng.feature_names)

        # DateTime
        dt_cols = col_types.get("datetime", [])
        dt_cols = [c for c in dt_cols if c in X_train.columns]
        if dt_cols:
            self._dt_eng = DateTimeFeatureEngineer(self.cfg.features.datetime)
            arr_tr = self._dt_eng.fit_transform(X_train[dt_cols])
            arr_te = self._dt_eng.transform(X_test[dt_cols])
            parts_train.append(arr_tr)
            parts_test.append(arr_te)
            all_names.extend(self._dt_eng.feature_names)

        # Text
        text_cols = col_types.get("text", [])
        text_cols = [c for c in text_cols if c in X_train.columns]
        if text_cols and getattr(self.cfg.features.text, "enabled", True):
            self._text_eng = TextFeatureEngineer(self.cfg.features.text)
            arr_tr = self._text_eng.fit_transform(X_train[text_cols])
            arr_te = self._text_eng.transform(X_test[text_cols])
            parts_train.append(arr_tr)
            parts_test.append(arr_te)
            all_names.extend(self._text_eng.feature_names)

        # Concat
        if not parts_train:
            raise ValueError("No features generated – check column type detection.")
        X_train_arr = np.hstack(parts_train)
        X_test_arr = np.hstack(parts_test)

        # Feature selection
        self._selector = FeatureSelector(self.cfg.features.selection)
        X_train_arr = self._selector.fit_transform(X_train_arr, y_train, all_names)
        X_test_arr = self._selector.transform(X_test_arr)
        selected_names = self._selector.selected_feature_names_

        logger.info(f"  Features: {len(all_names)} → {len(selected_names)} (after selection)")
        return X_train_arr, X_test_arr, selected_names

    def _transform_features(self, X: pd.DataFrame, col_types: dict) -> np.ndarray:
        """Apply fitted transforms for prediction."""
        parts: list = []
        num_cols = [c for c in col_types.get("numeric", []) if c in X.columns]
        if num_cols and self._numeric_eng:
            parts.append(self._numeric_eng.transform(X[num_cols]))
        cat_cols = [c for c in (col_types.get("categorical", []) + col_types.get("boolean", [])) if c in X.columns]
        if cat_cols and self._cat_eng:
            parts.append(self._cat_eng.transform(X[cat_cols]))
        dt_cols = [c for c in col_types.get("datetime", []) if c in X.columns]
        if dt_cols and self._dt_eng:
            parts.append(self._dt_eng.transform(X[dt_cols]))
        text_cols = [c for c in col_types.get("text", []) if c in X.columns]
        if text_cols and self._text_eng:
            parts.append(self._text_eng.transform(X[text_cols]))
        if not parts:
            raise ValueError("No features could be extracted from new data.")
        X_arr = np.hstack(parts)
        X_arr = self._selector.transform(X_arr)
        return X_arr
