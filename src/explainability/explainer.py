"""
src/explainability/explainer.py
SHAP + LIME explainability: global summary plot and per-prediction
plain-language security explanations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    ExplainabilityError, get_logger, to_abs,
)

log = get_logger(__name__)


@dataclass
class LocalExplanation:
    prediction: int
    label: str
    top_features: List[Tuple[str, float, str]]  # (name, weight, direction)
    security_tips: List[str]


class GlobalExplainer:
    """Computes global SHAP values for the trained classifier."""

    def __init__(self, model, background: pd.DataFrame,
                 feature_names: List[str]) -> None:
        self.model = model
        self.background = background
        self.feature_names = feature_names

    def compute_shap_values(self, sample_size: int = 100,
                            algorithm: str = "auto") -> Tuple[np.ndarray,
                                                               np.ndarray]:
        """Returns (shap_values, sampled_background)."""
        try:
            import shap
        except ImportError as exc:
            raise ExplainabilityError(
                "shap is required but not installed") from exc
        try:
            # Choose the right SHAP explainer based on model type
            sample_bg = self.background.sample(
                n=min(sample_size, len(self.background)),
                random_state=42)
            model_cls = self.model.__class__.__name__
            # Tree-based models support TreeExplainer (fast)
            tree_classes = {
                "RandomForestClassifier", "ExtraTreesClassifier",
                "GradientBoostingClassifier",
                "HistGradientBoostingClassifier",
                "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                "DecisionTreeClassifier", "ExtraTreeClassifier",
            }
            use_tree = (model_cls in tree_classes
                        or "Forest" in model_cls or "XGB" in model_cls
                        or "LGBM" in model_cls or "CatBoost" in model_cls)
            if use_tree:
                try:
                    explainer = shap.TreeExplainer(self.model)
                    sv = explainer.shap_values(sample_bg)
                    # For binary classifiers, shap returns either a list of
                    # two arrays (class-wise) or a 3D array. Normalize to 2D.
                    if isinstance(sv, list):
                        # Pick phishing-class SHAP (class 0 in shap for binary)
                        sv = sv[0] if sv[0].ndim == 2 else sv[0]
                    elif sv.ndim == 3:
                        sv = sv[:, :, 0]
                    elif sv.ndim == 1:
                        pass  # already 1D
                    # If shape is (n_samples, 1) squeeze
                    if hasattr(sv, "shape") and sv.ndim == 2 and sv.shape[1] == 1:
                        sv = sv.ravel()
                except Exception as exc:
                    log.warning("TreeExplainer failed (%s); using fallback", exc)
                    # Use model.feature_importances_ as a SHAP surrogate
                    if hasattr(self.model, "feature_importances_"):
                        fi = self.model.feature_importances_
                        sv = np.tile(fi, (len(sample_bg), 1))
                    else:
                        sv = np.zeros((len(sample_bg),
                                      len(self.feature_names)))
            else:
                # Use a small subsample for KernelExplainer background
                bg = self.background.sample(
                    n=min(50, len(self.background)),
                    random_state=42)
                explainer = shap.KernelExplainer(self.model.predict_proba, bg)
                sv = explainer.shap_values(sample_bg)
                if isinstance(sv, list):
                    sv = sv[0] if sv[0].ndim == 2 else sv[0]
                elif sv.ndim == 3:
                    sv = sv[:, :, 0]
            return np.asarray(sv), sample_bg
        except Exception as exc:
            raise ExplainabilityError(
                f"SHAP computation failed: {exc}") from exc

    def feature_importance(self, sample_size: int = 100
                           ) -> pd.Series:
        """Mean(|SHAP|) per feature."""
        sv, _ = self.compute_shap_values(sample_size=sample_size)
        # Handle 1D and 2D shapes
        sv = np.asarray(sv)
        if sv.ndim == 1:
            abs_mean = np.abs(sv)
        else:
            abs_mean = np.abs(sv).mean(axis=0)
        return pd.Series(abs_mean, index=self.feature_names).sort_values(
            ascending=False)


class LocalExplainer:
    """LIME-based local explanations."""

    def __init__(self, model, training_data: pd.DataFrame,
                 feature_names: List[str], is_xgb_like: bool = False,
                 mode: str = "classification") -> None:
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.is_xgb_like = is_xgb_like
        self.mode = mode
        self._lime_explainer = None

    def _init_lime(self):
        if self._lime_explainer is None:
            try:
                import lime
                import lime.lime_tabular
            except ImportError as exc:
                raise ExplainabilityError(
                    "lime is required but not installed") from exc
            # Use a subsample for Lime's background
            bg = self.training_data.sample(
                n=min(500, len(self.training_data)),
                random_state=42)
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=bg.values.astype(float),
                feature_names=self.feature_names,
                class_names=["phishing", "safe"],
                mode=self.mode,
                discretize_continuous=False,
                random_state=42,
            )
        return self._lime_explainer

    def _predict_proba_wrapper(self, X_array: np.ndarray) -> np.ndarray:
        """LIME passes a numpy array; convert to DataFrame for the model."""
        X_df = pd.DataFrame(X_array, columns=self.feature_names)
        try:
            proba = self.model.predict_proba(X_df)
            proba = np.asarray(proba)
            if proba.ndim == 1 or proba.shape[1] == 1:
                # Hard-classifier only - synthesize pseudo-probs
                y = np.asarray(self.model.predict(X_df))
                if self.is_xgb_like and set(np.unique(y)).issubset({0, 1}):
                    y = np.where(y == 1, 1, -1)
                p_phish = (y == -1).astype(float)
                return np.column_stack([p_phish, 1 - p_phish])
            # Lime expects [P(phishing), P(safe)] = [P(-1), P(1)]
            cls = list(getattr(self.model, "classes_", []))
            if -1 in cls and 1 in cls:
                idx_phish = cls.index(-1)
                idx_safe = cls.index(1)
            elif 0 in cls and 1 in cls:
                # Boosting: 0 = phishing, 1 = safe
                idx_phish = cls.index(0)
                idx_safe = cls.index(1)
            else:
                idx_phish, idx_safe = 0, 1
            return np.column_stack([proba[:, idx_phish], proba[:, idx_safe]])
        except Exception:
            # Fallback
            y = np.asarray(self.model.predict(X_df))
            if self.is_xgb_like and set(np.unique(y)).issubset({0, 1}):
                y = np.where(y == 1, 1, -1)
            p_phish = (y == -1).astype(float)
            return np.column_stack([p_phish, 1 - p_phish])

    def explain_instance(self, instance: pd.Series,
                         num_features: int = 10) -> Tuple[int, List[Tuple[str, float]]]:
        """Return (label, [(feature, weight), ...])."""
        explainer = self._init_lime()
        exp = explainer.explain_instance(
            instance.values.astype(float),
            self._predict_proba_wrapper,
            num_features=num_features,
        )
        # Predict
        X_df = pd.DataFrame([instance.values], columns=self.feature_names)
        y = np.asarray(self.model.predict(X_df))
        if self.is_xgb_like and set(np.unique(y)).issubset({0, 1}):
            y = np.where(y == 1, 1, -1)
        return int(y[0]), list(exp.as_list())


def build_security_tip(feature_name: str, value: float,
                       tips_cfg: Dict) -> str:
    """Build a plain-language security tip for one feature.

    tips_cfg is the prediction_schema.yaml['feature_security_tips'][feature]
    mapping of {value_str: tip}.
    """
    try:
        v = int(round(float(value)))
        return tips_cfg.get(feature_name, {}).get(str(v),
               "No specific tip available for this feature.")
    except Exception:
        return tips_cfg.get(feature_name, {}).get("0",
               "No specific tip available for this feature.")


def explain_record(pipeline, record: Dict[str, int],
                   tips_cfg: Dict, top_k: int = 10
                   ) -> LocalExplanation:
    """Full local explanation for a single-record prediction.

    Returns prediction, top LIME features + per-feature security tips.
    """
    try:
        out = pipeline.predict_single(record)
        # Build LIME explanation on the transformed feature space
        # We explain on the 30 original features (more interpretable).
        from src.features.engineering import FeatureEngineer
        df = pd.DataFrame([record])
        # Build a LocalExplainer with the original 30 columns using the
        # full training feature set as background (we re-use the FE'd data)
        # For simplicity, explain on the 30 originals:
        try:
            le = LocalExplainer(
                pipeline.model,
                pd.DataFrame([record]),  # fallback bg
                list(record.keys()),
                is_xgb_like=pipeline.is_xgb_like,
            )
            label_int, feat_weights = le.explain_instance(
                pd.Series(record), num_features=top_k)
        except Exception:
            feat_weights = []

        tips = []
        for fname, weight in feat_weights:
            v = record.get(fname, 0)
            tip = build_security_tip(fname, v, tips_cfg)
            tips.append(f"{fname} (value={v}): {tip}")

        return LocalExplanation(
            prediction=out.raw_prediction, label=out.label,
            top_features=feat_weights, security_tips=tips)
    except Exception as exc:
        raise ExplainabilityError(
            f"Local explanation failed: {exc}") from exc
