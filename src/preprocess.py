"""
Preprocessing pipeline for the phishing URL classifier.

- Encodes `tld` using top-N frequency bucketing ("other" for rare TLDs).
- Drops string columns (`url`, `dom`) that the model cannot consume directly.
- Provides a single fit/transform interface that is saved alongside the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Top-N most-frequent TLDs that get their own one-hot slot.  Everything
# else collapses into "other".  50 covers ~95% of mass in this dataset.
DEFAULT_TOP_N_TLDS: int = 50


@dataclass
class Preprocessor(BaseEstimator, TransformerMixin):
    """Fit/transform URL features into a model-ready numeric matrix.

    Attributes
    ----------
    top_tlds : list of str
        TLDs that get their own binary column after `fit`.
    numeric_features : list of str
        Numeric columns passed through untouched.
    """

    top_tlds: List[str] = field(default_factory=list)
    numeric_features: List[str] = field(default_factory=list)
    top_n: int = DEFAULT_TOP_N_TLDS

    def fit(self, X: pd.DataFrame, y=None) -> "Preprocessor":  # noqa: D401
        # Discover which numeric columns exist in the input.
        self.numeric_features = [
            c for c in X.columns
            if c != "tld" and X[c].dtype.kind in {"i", "u", "f"}
        ]

        # Pick the top-N most-frequent TLDs.
        if "tld" in X.columns:
            counts = X["tld"].fillna("").astype(str).value_counts()
            self.top_tlds = counts.head(self.top_n).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        # Bucket TLDs.
        if "tld" in out.columns:
            tld_normalized = out["tld"].fillna("").astype(str)
            out["tld_bucketed"] = tld_normalize_tld(tld_normalized, self.top_tlds)
            # One-hot the bucketed TLD.
            dummies = pd.get_dummies(out["tld_bucketed"], prefix="tld")
            out = pd.concat([out.drop(columns=["tld", "tld_bucketed"]), dummies], axis=1)
        else:
            # No TLD column -> still emit the learned dummy columns (all zeros).
            for tld in self.top_tlds:
                out[f"tld_{tld}"] = 0

        # Ensure every expected dummy column exists (training/inference parity).
        expected_dummy_cols = [f"tld_{t}" for t in self.top_tlds] + ["tld_other"]
        for col in expected_dummy_cols:
            if col not in out.columns:
                out[col] = 0

        # Drop any non-numeric columns we don't want the model to see.
        drop_cols = [c for c in ("url", "dom") if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        # Keep only numeric + dummy columns.
        keep = self.numeric_features + expected_dummy_cols
        keep = [c for c in keep if c in out.columns]
        out = out[keep]
        return out

    def feature_names(self) -> List[str]:
        """Return the post-transform feature order."""
        return self.numeric_features + [f"tld_{t}" for t in self.top_tlds] + ["tld_other"]


def tld_normalize_tld(series: pd.Series, top_tlds: Sequence[str]) -> pd.Series:
    """Map any TLD not in `top_tlds` to the literal string "other"."""
    return series.where(series.isin(top_tlds), other="other")
