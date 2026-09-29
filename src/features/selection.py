"""
src/features/selection.py
Feature selection: ranks features by mutual information, model-based
importance, and RFE; selects the union of the top-K.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif, RFE, SelectFromModel,
)
from sklearn.linear_model import LogisticRegression

from src.utils import (
    FeatureSelectionError, get_logger, save_joblib, to_abs, write_json,
)

log = get_logger(__name__)


@dataclass
class SelectionReport:
    mi_scores: dict
    rf_importances: dict
    rfe_ranking: dict
    selected_features: list


class FeatureSelector:
    """Selects the most discriminative features for classification.

    The original 30 ternary features are ALWAYS retained (they are the
    dataset's identity, expected by the schema). Engineered features are
    layered on top via mutual_information + model-based + RFE.
    """

    def __init__(self, original_features: List[str], top_k: int = 15,
                 keep_original: bool = True,
                 random_state: int = 42) -> None:
        self.original_features = list(original_features)
        self.top_k = top_k
        self.keep_original = keep_original
        self.random_state = random_state

    def compute_mi(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        log.info("Computing mutual information scores...")
        mi = mutual_info_classif(
            X, y, random_state=self.random_state, discrete_features=True
        )
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)

    def compute_rf_importance(self, X: pd.DataFrame,
                              y: pd.Series) -> pd.Series:
        log.info("Computing RandomForest feature importances...")
        rf = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state,
            class_weight="balanced", n_jobs=-1,
        )
        rf.fit(X, y)
        return pd.Series(rf.feature_importances_,
                         index=X.columns).sort_values(ascending=False)

    def compute_rfe_ranking(self, X: pd.DataFrame,
                            y: pd.Series, n_to_select: int) -> pd.Series:
        """RFE ranking using a fast L1-penalized LogisticRegression."""
        log.info("Computing RFE ranking (n_to_select=%d)...", n_to_select)
        # Use a fast L1 estimator + cap iterations so RFE runs quickly
        estimator = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=200,
            class_weight="balanced", random_state=self.random_state,
        )
        # Subsample to 2k rows for speed - RFE is O(features^2 * fits)
        if X.shape[0] > 2000:
            idx = np.random.default_rng(self.random_state).choice(
                X.shape[0], 2000, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]
        n_sel = min(n_to_select, X.shape[1] - 1)
        try:
            rfe = RFE(estimator, n_features_to_select=n_sel, step=0.2)
            rfe.fit(X, y)
            ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
        except Exception as exc:
            log.warning("RFE failed (%s); using SelectFromModel fallback", exc)
            sfm = SelectFromModel(estimator, max_features=n_sel,
                                  threshold=-np.inf)
            sfm.fit(X, y)
            # Selected -> rank 1, others -> rank 2 in original order
            ranking = pd.Series(
                [1 if s else 2 for s in sfm.get_support()],
                index=X.columns).sort_values()
        return ranking

    def select(self, X: pd.DataFrame, y: pd.Series,
               out_dir: str | Path) -> tuple[pd.DataFrame, SelectionReport]:
        try:
            out = to_abs(out_dir)
            out.mkdir(parents=True, exist_ok=True)

            mi = self.compute_mi(X, y)
            rf_imp = self.compute_rf_importance(X, y)
            rfe_rank = self.compute_rfe_ranking(
                X, y, n_to_select=min(self.top_k, X.shape[1] - 1))

            # Union of top-K from each method (plus all originals)
            top_mi = set(mi.head(self.top_k).index)
            top_rf = set(rf_imp.head(self.top_k).index)
            top_rfe = set(rfe_rank.head(self.top_k).index)
            union = top_mi | top_rf | top_rfe
            if self.keep_original:
                union |= set(self.original_features)

            # Preserve canonical ordering: originals first, then engineered
            selected = [c for c in X.columns if c in union]
            self.selected_features_ = selected

            # Persist selector + report
            save_joblib(self, out / "feature_selector.joblib")
            report = SelectionReport(
                mi_scores=mi.to_dict(),
                rf_importances=rf_imp.to_dict(),
                rfe_ranking=rfe_rank.to_dict(),
                selected_features=selected,
            )
            write_json(out / "selection_report.json", {
                "mi_scores": report.mi_scores,
                "rf_importances": report.rf_importances,
                "rfe_ranking": report.rfe_ranking,
                "selected_features": report.selected_features,
                "n_selected": len(selected),
            })
            log.info("Selected %d / %d features (union of MI, RF, RFE, "
                     "+originals)", len(selected), X.shape[1])
            log.info("Top-5 MI: %s", list(mi.head(5).items()))
            return X[selected], report
        except Exception as exc:
            raise FeatureSelectionError(
                f"Feature selection failed: {exc}") from exc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise FeatureSelectionError("Selector not fit; call select() first")
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            # Fill missing engineered columns with 0 (shouldn't happen if
            # the upstream feature engineer was applied)
            for c in missing:
                X = X.copy()
                X[c] = 0
        return X[self.selected_features_]
