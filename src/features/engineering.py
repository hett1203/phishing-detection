"""
src/features/engineering.py
Feature engineering: builds interpretable interaction features and
optional non-linear transformations on top of the original 30 ternary features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src.utils import (
    TransformationError, get_logger, save_joblib, to_abs,
)

log = get_logger(__name__)


@dataclass
class FeatureEngineer:
    """Adds interpretable engineered features on top of the original 30.

    Engineered features:
      * ``risk_score_sum``        -- sum of all feature values (high = legit,
                                      low = phishing)
      * ``risk_score_mean``
      * ``risk_score_std``
      * ``n_suspicious``           -- count of features == -1
      * ``n_legit``                -- count of features == +1
      * ``n_neutral``              -- count of features == 0
      * ``phishing_ratio``         -- n_suspicious / 30
      * ``legit_ratio``            -- n_legit / 30
      * Top-K pairwise interaction products (by mutual information, selected
        upstream -- fallback to the 8 strongest known phishing drivers)
    """
    features: List[str]
    top_k_interactions: int = 8
    interaction_features: bool = True

    STRONG_DRIVERS = [
        "SSLfinal_State", "URL_of_Anchor", "web_traffic", "having_Sub_Domain",
        "Prefix_Suffix", "Domain_registeration_length", "Page_Rank",
        "Links_pointing_to_page",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None,
            mi_scores: pd.Series | None = None) -> "FeatureEngineer":
        """Decide which pairs to use for interactions.

        If ``mi_scores`` is provided (a Series indexed by feature name),
        use the top-K features by MI. Otherwise fall back to the
        known-strong list. ``fit`` is idempotent; it only records the
        selected interaction pairs.
        """
        if not self.interaction_features:
            self.interaction_pairs_: List[Tuple[str, str]] = []
            return self

        if mi_scores is not None:
            avail = [f for f in mi_scores.index if f in self.features]
            top_k = list(mi_scores.loc[avail].sort_values(ascending=False)
                         .head(self.top_k_interactions).index)
        else:
            top_k = [f for f in self.STRONG_DRIVERS if f in self.features]
            # Pad if too few
            if len(top_k) < self.top_k_interactions:
                for f in self.features:
                    if f not in top_k:
                        top_k.append(f)
                    if len(top_k) == self.top_k_interactions:
                        break
            top_k = top_k[: self.top_k_interactions]

        # All unordered pairs among the top-K
        pairs = []
        for i, a in enumerate(top_k):
            for b in top_k[i + 1:]:
                pairs.append((a, b))
        self.interaction_pairs_ = pairs
        self.engineered_names_ = self._engineered_names()
        log.info("Selected %d interaction pairs (top-K=%d)",
                 len(pairs), self.top_k_interactions)
        return self

    def _engineered_names(self) -> List[str]:
        names = [
            "risk_score_sum", "risk_score_mean", "risk_score_std",
            "n_suspicious", "n_legit", "n_neutral",
            "phishing_ratio", "legit_ratio",
        ]
        for a, b in self.interaction_pairs_:
            names.append(f"i__{a}__{b}")
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "engineered_names_"):
            self.fit(X)  # idempotent fallback

        out = X.copy()
        # Statistical aggregates
        arr = X.values
        out["risk_score_sum"] = arr.sum(axis=1)
        out["risk_score_mean"] = arr.mean(axis=1)
        out["risk_score_std"] = arr.std(axis=1) if arr.shape[1] > 1 else 0.0
        out["n_suspicious"] = (X == -1).sum(axis=1).astype(np.int8)
        out["n_legit"] = (X == 1).sum(axis=1).astype(np.int8)
        out["n_neutral"] = (X == 0).sum(axis=1).astype(np.int8)
        out["phishing_ratio"] = out["n_suspicious"] / float(X.shape[1])
        out["legit_ratio"] = out["n_legit"] / float(X.shape[1])
        # Interaction products
        for a, b in self.interaction_pairs_:
            if a in X.columns and b in X.columns:
                out[f"i__{a}__{b}"] = X[a] * X[b]
            else:
                out[f"i__{a}__{b}"] = 0
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None,
                     mi_scores: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X, y, mi_scores)
        return self.transform(X)

    def save(self, path: str | Path) -> Path:
        return save_joblib(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEngineer":
        from src.utils import load_joblib
        return load_joblib(path)


def fit_engineer(X: pd.DataFrame, features: List[str],
                 mi_scores: pd.Series | None, params_cfg,
                 out_dir: str | Path) -> Tuple[pd.DataFrame, FeatureEngineer]:
    try:
        out = to_abs(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        eng_cfg = params_cfg.feature_engineering
        eng = FeatureEngineer(
            features=features,
            top_k_interactions=int(eng_cfg.top_k_interactions),
            interaction_features=bool(eng_cfg.interaction_features),
        )
        X_eng = eng.fit_transform(X, None, mi_scores)
        eng.save(out / "feature_engineer.joblib")
        log.info("Feature engineering: %d original -> %d total features",
                 X.shape[1], X_eng.shape[1])
        return X_eng, eng
    except Exception as exc:
        raise TransformationError(
            f"Feature engineering failed: {exc}") from exc
