"""Prediction helpers shared by all Streamlit pages."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

# Resolve paths relative to project root (parent of `src/`).
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"
SHAP_DIR = ARTIFACTS / "shap"

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single-URL prediction."""
    label: str                  # "Phishing" or "Safe"
    probability: float          # phishing-class probability in [0, 1]
    is_phishing: bool
    risk_score: int             # 0-100 risk gauge value
    top_features: list          # list of (feature_name, contribution) tuples


class Predictor:
    """Lazy-loaded wrapper around the saved best model + preprocessor."""

    _instance: "Predictor | None" = None

    def __init__(self) -> None:
        self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        try:
            self.model = joblib.load(MODELS_DIR / "best_model.joblib")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The saved model was created with an incompatible scikit-learn "
                "version. Rebuild it with `python src/train.py` in this environment."
            ) from exc
        with open(METRICS_DIR / "best_meta.json") as f:
            self.meta = json.load(f)
        self.feature_names: list[str] = list(self.meta["feature_columns"])

    @classmethod
    def get(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---- Single-row inference ----
    def predict_features_df(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return phishing-class probabilities for the given feature rows."""
        transformed = self.preprocessor.transform(features_df)
        # Reorder to match training
        transformed = transformed.reindex(columns=self.feature_names, fill_value=0)
        proba = self.model.predict_proba(transformed)
        return proba[:, 1]

    # ---- Per-feature contributions for "why this was flagged" ----
    def contributions_for(self, features_df: pd.DataFrame) -> list[tuple[str, float]]:
        """Return a list of (feature, contribution) tuples for the first row.

        Uses simple standardized deviation from the training mean as a
        cheap, robust signal that doesn't require loading SHAP at runtime.
        """
        transformed = self.preprocessor.transform(features_df).iloc[0]
        # For each numeric feature, the "contribution" is the deviation from
        # 0 (since features are mostly counts) normalized by feature std.
        contribs: list[tuple[str, float]] = []
        for col in transformed.index:
            val = float(transformed[col])
            if col.startswith("tld_"):
                if val > 0:
                    contribs.append((col, 0.5))
            else:
                # Larger absolute value = larger signal in this count-based feature set.
                contribs.append((col, abs(val)))
        # Sort by magnitude descending
        contribs.sort(key=lambda x: x[1], reverse=True)
        return contribs


# ---- Security tips per feature ----
SECURITY_TIPS: dict[str, str] = {
    "is_ip": "This site uses an IP address instead of a domain name — a common phishing indicator.",
    "url_len": "The URL is unusually long — long URLs are often used to hide redirects or payloads.",
    "is_https": "This site does not use HTTPS — modern legitimate sites almost always encrypt traffic.",
    "qm_cnt": "An unusually high number of '?' characters in the URL — often indicates many query parameters carrying malicious payloads.",
    "amp_cnt": "Many '&' characters in the URL — heavy query strings are common in phishing redirects.",
    "eq_cnt": "Many '=' characters in the URL — used to inject script payloads or session hijack tokens.",
    "dot_cnt": "Many dots in the URL — sub-domain abuse is a classic obfuscation technique.",
    "dash_cnt": "Many dashes in the URL — often used to mimic legitimate brand names (e.g., 'paypal-secure-login').",
    "under_cnt": "Many underscores in the URL — uncommon in legitimate domains, often used in obfuscated URLs.",
    "special_cnt": "High number of special characters — typically indicates encoded payloads.",
    "digit_cnt": "High number of digits in the URL — phishing URLs frequently embed numbers to evade filters.",
    "subdom_cnt": "Many sub-domains — attackers stack sub-domains to mimic legitimate organizations.",
    "tld_other": "An unusual or rare top-level domain — many phishing sites use obscure TLDs.",
    "slash_cnt": "Many slashes in the URL — long redirect chains are a hallmark of phishing links.",
    "path_len": "An unusually long path — phishing pages often hide deep inside directory structures.",
    "query_len": "Long query string — frequently used to carry attack payloads.",
    "entropy": "High URL entropy — random-looking character distribution is a known obfuscation signal.",
    "letter_ratio": "Unusual letter ratio — extreme values often indicate obfuscated URLs.",
    "digit_ratio": "High digit ratio — URLs heavy in digits are often machine-generated phishing links.",
    "spec_ratio": "High special-character ratio — encoded payloads and obfuscation.",
}


def get_security_tip(feature: str) -> str:
    """Return a plain-language tip for the given feature, or a default."""
    return SECURITY_TIPS.get(feature, f"Unusual value in '{feature}' — review before trusting this site.")


def load_metrics() -> pd.DataFrame:
    """Load the model comparison DataFrame."""
    return pd.read_csv(METRICS_DIR / "model_comparison.csv")


def load_best_meta() -> dict:
    """Load the best-model metadata."""
    with open(METRICS_DIR / "best_meta.json") as f:
        return json.load(f)


def load_roc_curves() -> dict:
    with open(METRICS_DIR / "roc_curves.json") as f:
        return json.load(f)


def load_confusion_matrices() -> dict:
    with open(METRICS_DIR / "confusion_matrices.json") as f:
        return json.load(f)


def load_shap_artifacts() -> tuple[np.ndarray, pd.DataFrame]:
    """Return (shap_values, sample_X) for global explainability plots."""
    sv = np.load(SHAP_DIR / "shap_values.npy")
    sample = pd.read_csv(SHAP_DIR / "shap_sample.csv")
    return sv, sample
