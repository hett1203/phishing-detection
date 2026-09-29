"""
src/prediction/pipeline.py
Prediction pipeline: single-record and batch inference. Reproduces the exact
schema of predicted_file.csv (30 features + 'Result' string label +
confidence + probability_phishing).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils import (
    get_logger, load_joblib, map_numeric_to_label,
    to_abs, write_json,
)

log = get_logger(__name__)


@dataclass
class PredictionOutput:
    label: str            # "phising" or "safe"
    confidence: float     # max-class probability [0, 1]
    p_phishing: float
    p_safe: float
    raw_prediction: int    # -1 or 1


def _phish_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(class = phishing) for each row in X."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        cls = list(getattr(model, "classes_", []))
        if -1 in cls:
            idx = cls.index(-1)
        elif 0 in cls:
            idx = cls.index(0)
        else:
            idx = 0
        return np.asarray(proba)[:, idx]
    # No probabilities: fall back to (y == -1)
    y = np.asarray(model.predict(X))
    # Boosting libs output {0, 1}; remap
    if set(np.unique(y)).issubset({0, 1}):
        return y.astype(float)
    return (y == -1).astype(float)


def _binary_predict(model, X: pd.DataFrame, is_xgb_like: bool) -> np.ndarray:
    """Predict in {-1, 1}."""
    raw = np.asarray(model.predict(X))
    if is_xgb_like and set(np.unique(raw)).issubset({0, 1}):
        return np.where(raw == 1, 1, -1)
    return raw


class PredictionPipeline:
    """End-to-end prediction: feature-engineering -> selection -> model.

    Args:
        feature_engineer: fit FeatureEngineer object
        feature_selector: fit FeatureSelector object
        transformer: optional DataTransformer (PCA pipeline)
        model: trained classifier
        is_xgb_like: True for XGBoost / LightGBM / CatBoost (label remap)
    """

    def __init__(self, feature_engineer, feature_selector,
                 model, is_xgb_like: bool = False,
                 transformer=None) -> None:
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.transformer = transformer
        self.model = model
        self.is_xgb_like = is_xgb_like

    # -------- Single-record prediction -------- #
    def predict_single(self, record: Dict[str, int]
                       ) -> PredictionOutput:
        """Predict for a single 30-feature record."""
        df = pd.DataFrame([record])
        df = self.feature_engineer.transform(df)
        df = self.feature_selector.transform(df)
        if self.transformer is not None:
            df = pd.DataFrame(
                self.transformer.transform(df),
                columns=self.transformer.named_steps.get(
                    "pca", None) and
                [f"PC{i+1}" for i in range(self.transformer["pca"].n_components_)]
                or df.columns,
                index=df.index) if False else df  # passthrough-safe

        p_phish = float(_phish_probability(self.model, df)[0])
        p_safe = 1.0 - p_phish
        if p_phish >= 0.5:
            label = map_numeric_to_label(-1)
            raw = -1
            conf = p_phish
        else:
            label = map_numeric_to_label(1)
            raw = 1
            conf = p_safe
        return PredictionOutput(
            label=label, confidence=conf,
            p_phishing=p_phish, p_safe=p_safe, raw_prediction=raw)

    # -------- Batch prediction -------- #
    def predict_batch(self, X: pd.DataFrame,
                      out_path: Optional[str | Path] = None
                      ) -> pd.DataFrame:
        """Predict for an unlabeled batch CSV.

        Output schema matches ``predicted_file.csv`` (the reference file
        provided by the UCI dataset): the 30 original features, plus a
        'Result' column with the string labels ``'phising'`` / ``'safe'``,
        plus a 'confidence' and 'probability_phishing' column.
        """
        original = X.copy()
        feats = self.feature_engineer.transform(original)
        feats = self.feature_selector.transform(feats)
        raw = _binary_predict(self.model, feats, self.is_xgb_like)
        p_phish = _phish_probability(self.model, feats)

        labels = np.where(raw == -1, "phising", "safe")
        confidence = np.where(raw == -1, p_phish, 1.0 - p_phish)

        # Output: original 30 features + Result + confidence + p_phishing
        out = original.copy()
        out["Result"] = labels
        out["confidence"] = confidence
        out["probability_phishing"] = p_phish
        if out_path is not None:
            p = to_abs(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(p, index=False)
            log.info("Wrote batch predictions -> %s  (rows=%d)",
                     p, len(out))
        return out

    # -------- Batch prediction from CSV file -------- #
    def predict_csv(self, csv_path: str | Path,
                    out_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(to_abs(csv_path))
        return self.predict_batch(df, out_path=out_path)


def load_pipeline(trained_models_dir: str | Path, model_name: str,
                  preprocessing_dir: str | Path,
                  is_xgb_like: bool = False) -> PredictionPipeline:
    """Convenience loader: builds a PredictionPipeline from artifacts."""
    tm = to_abs(trained_models_dir)
    pp = to_abs(preprocessing_dir)
    # Locate model file
    candidates = [
        tm / f"{model_name.lower()}.joblib",
        tm / model_name.replace(" ", "_").lower() + ".joblib",
        tm / f"best_model.joblib",
    ]
    model_path = next((c for c in candidates if c.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"Could not find model file for {model_name} in {tm}")
    model = load_joblib(model_path)
    fe = load_joblib(pp / "feature_engineer.joblib")
    fs = load_joblib(pp / "feature_selector.joblib")
    return PredictionPipeline(
        feature_engineer=fe, feature_selector=fs,
        model=model, is_xgb_like=is_xgb_like)
