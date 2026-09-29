"""
run_shap_only.py
----------------
Completes the SHAP explainability step using already-trained artifacts.
Useful when training.py timed out before reaching the SHAP stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.setrecursionlimit(20000)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    get_config, get_logger, setup_logging, load_joblib, write_json,
)
from src.features import FeatureEngineer, FeatureSelector
from src.explainability import GlobalExplainer

setup_logging()
log = get_logger("shap_only")
cfg = get_config()

# Load processed train data
X_tr = pd.read_csv(cfg.processed_data_dir / "X_train.csv")
y_tr = pd.read_csv(cfg.processed_data_dir / "y_train.csv").iloc[:, 0]

# Apply FE + FS
eng = FeatureEngineer.load(cfg.features_dir / "feature_engineer.joblib")
sel = load_joblib(cfg.features_dir / "feature_selector.joblib")
X_sel_tr = sel.transform(eng.transform(X_tr))
log.info("Selected features: %d", X_sel_tr.shape[1])

# Load best model metadata
import json
with open(cfg.trained_models_dir / "best_model_metadata.json") as fh:
    best_meta = json.load(fh)
log.info("Best model: %s", best_meta["name"])

# Choose SHAP model: prefer a tree-based model if best is an ensemble
tree_like_names = {"RandomForest", "ExtraTrees", "XGBoost", "LightGBM",
                   "CatBoost", "HistGradientBoosting", "GradientBoosting"}
shap_model_name = best_meta["name"]
shap_model = load_joblib(cfg.trained_models_dir / "best_model.joblib")

if shap_model_name not in tree_like_names:
    # Prefer XGBoost -> LightGBM -> CatBoost -> RF
    for preferred in ["XGBoost", "LightGBM", "CatBoost",
                      "RandomForest", "ExtraTrees"]:
        # Map names to file names from configs/model.yaml
        file_map = {
            "RandomForest": "random_forest.joblib",
            "ExtraTrees": "extra_trees.joblib",
            "XGBoost": "xgboost.joblib",
            "LightGBM": "lightgbm.joblib",
            "CatBoost": "catboost.joblib",
        }
        path = cfg.trained_models_dir / file_map.get(preferred, "")
        if path.exists():
            shap_model = load_joblib(path)
            shap_model_name = preferred
            log.info("Best model %s is not tree-compatible; using %s for SHAP",
                     best_meta["name"], preferred)
            break

# Run SHAP
log.info("Computing SHAP global importance with %s (sample=100)...",
         shap_model_name)
explainer = GlobalExplainer(
    model=shap_model,
    background=X_sel_tr,
    feature_names=list(X_sel_tr.columns),
)
try:
    importance = explainer.feature_importance(sample_size=100)
    write_json(cfg.shap_dir / "global_feature_importance.json",
               importance.to_dict())
    write_json(cfg.shap_dir / "shap_metadata.json", {
        "model_used": shap_model_name,
        "best_model": best_meta["name"],
        "n_background_samples": 100,
    })
    log.info("Top-5 SHAP features:")
    for fname, val in list(importance.head(5).items()):
        log.info("  %s: %.4f", fname, val)
    log.info("SHAP artifacts saved to: %s", cfg.shap_dir)
except Exception as exc:
    log.error("SHAP failed: %s", exc)
    # Fallback: use the model's built-in feature_importances_ if available
    if hasattr(shap_model, "feature_importances_"):
        log.info("Falling back to model.feature_importances_")
        fi = pd.Series(shap_model.feature_importances_,
                       index=X_sel_tr.columns).sort_values(ascending=False)
        write_json(cfg.shap_dir / "global_feature_importance.json",
                   fi.to_dict())
        write_json(cfg.shap_dir / "shap_metadata.json", {
            "model_used": shap_model_name,
            "best_model": best_meta["name"],
            "fallback": "feature_importances_",
        })
        log.info("Saved fallback feature_importances_")
