"""
src/features/transformer.py
Data transformation layer: builds interaction/polynomial features and applies
optional PCA. Persisted as a sklearn Pipeline so the same transformation is
applied at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import (
    TransformationError, get_logger, save_joblib, to_abs,
)

log = get_logger(__name__)


@dataclass
class TransformationArtifact:
    transformer: Pipeline
    feature_names_in: List[str]
    feature_names_out: List[str]
    n_components_pca: int


class DataTransformer:
    """Builds and applies the transformation pipeline.

    Notes
    -----
    The UCI Phishing dataset is already ternary-encoded, so scaling is
    applied only when PCA is requested (PCA benefits from centered features).
    """

    def __init__(self, features: List[str], params_cfg) -> None:
        self.features = list(features)
        self.params = params_cfg
        self.fe_cfg = params_cfg.feature_engineering

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        steps = []
        if self.fe_cfg.apply_pca:
            # Center features before PCA (ternary -> zero mean)
            steps.append(("scaler", StandardScaler(with_std=False)))
            n_comp = min(self.fe_cfg.pca_components, X.shape[1])
            steps.append(("pca", PCA(n_components=n_comp,
                                     random_state=42)))
            log.info("Building pipeline: StandardScaler -> PCA(n=%d)", n_comp)
        else:
            steps.append(("passthrough", "passthrough"))
            log.info("Building pipeline: identity (no scaling/PCA - data is "
                     "ternary-encoded)")
        return Pipeline(steps)

    def fit_transform(self, X: pd.DataFrame, out_dir: str | Path
                      ) -> tuple[pd.DataFrame, TransformationArtifact]:
        try:
            out = to_abs(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            pipe = self.build_pipeline(X)
            Xt = pipe.fit_transform(X)
            if hasattr(pipe, "named_steps") and "pca" in pipe.named_steps:
                n_comp = pipe.named_steps["pca"].n_components_
                cols = [f"PC{i+1}" for i in range(n_comp)]
                # Use the original features when PCA disabled
                feature_names_out = cols
            else:
                feature_names_out = list(X.columns)
                n_comp = 0
            Xt_df = pd.DataFrame(Xt, columns=feature_names_out, index=X.index)
            art = TransformationArtifact(
                transformer=pipe,
                feature_names_in=list(X.columns),
                feature_names_out=feature_names_out,
                n_components_pca=n_comp,
            )
            save_joblib(art, out / "transformer.joblib")
            log.info("Transformed shape: %s -> %s", X.shape, Xt_df.shape)
            return Xt_df, art
        except Exception as exc:
            raise TransformationError(
                f"Transformation failed: {exc}") from exc

    def transform(self, X: pd.DataFrame, transformer: Pipeline
                  ) -> pd.DataFrame:
        try:
            Xt = transformer.transform(X)
            if hasattr(transformer, "named_steps") and \
                    "pca" in transformer.named_steps:
                n_comp = transformer.named_steps["pca"].n_components_
                cols = [f"PC{i+1}" for i in range(n_comp)]
            else:
                cols = list(X.columns)
            return pd.DataFrame(Xt, columns=cols, index=X.index)
        except Exception as exc:
            raise TransformationError(
                f"Transform (inference) failed: {exc}") from exc
