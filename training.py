#!/usr/bin/env python3
"""
training.py
Master orchestrator: runs the entire Phishing Website Detection pipeline.

Execution order:
  1. Setup logging + load configs
  2. Create artifact directories
  3. Ingest raw CSVs
  4. Validate against schema
  5. (Optional) PCA / scaling transformer
  6. Feature engineering (interactions + aggregates)
  7. Feature selection (MI + RF + RFE)
  8. Clustering (10 algorithms, optional/interpretive)
  9. Supervised classification training (9 models + 2 ensembles + AutoGluon)
 10. Evaluation + best-model selection
 11. SHAP / LIME explainability
 12. Save best model + metadata

Usage:
    python training.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Joblib serialization of large trees (Birch, RandomForest, etc.) can hit
# Python's default recursion limit (1000). Bump it early.
sys.setrecursionlimit(20000)

from src.utils import (
    ProjectConfig, get_config, get_logger, save_joblib, set_seed,
    to_abs, write_json, map_numeric_to_label,
)
from src.utils.logger import setup_logging
from src.data import DataIngestion, DataValidation
from src.features import (
    FeatureEngineer, FeatureSelector, DataTransformer, fit_engineer,
)
from src.clustering import ClusterTrainer, name_clusters
from src.models import ClassificationTrainer, tune_xgboost
from src.evaluation import evaluate_all, select_best
from src.explainability import GlobalExplainer
from src.prediction import PredictionPipeline


def main() -> None:
    t_start = time.time()
    setup_logging()
    log = get_logger("training")
    log.info("=" * 72)
    log.info("Phishing Website Detection - Training Pipeline")
    log.info("=" * 72)

    cfg: ProjectConfig = get_config()
    cfg.create_all_dirs()
    set_seed(cfg.random_state)

    # -------- 1. Ingest --------
    log.info("[1/11] Data Ingestion")
    ing = DataIngestion(cfg.raw_data_dir,
                       train_file=cfg.config.files.train_data,
                       test_file=cfg.config.files.test_data)
    ingest_art = ing.ingest()

    # -------- 2. Validate --------
    log.info("[2/11] Data Validation")
    validator = DataValidation(
        schema_cfg=cfg.schema,
        features=cfg.features,
        target_name=cfg.target_name,
    )
    validator.validate(ingest_art.train_df, ingest_art.test_df,
                       cfg.validated_data_dir)

    # -------- 3. Split train/val/test --------
    log.info("[3/11] Train/Test split")
    splits = cfg.params.splits
    train_df = ingest_art.train_df.copy()
    y = train_df[cfg.target_name].astype(int)
    X = train_df[cfg.features].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=float(splits.test_size),
        random_state=int(splits.random_state),
        stratify=y if splits.stratify else None,
    )
    # Validate-set from train
    val_frac = float(splits.val_size) / (1.0 - float(splits.test_size))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        random_state=int(splits.random_state),
        stratify=y_trainval if splits.stratify else None,
    )
    log.info("Train rows=%d Val rows=%d Test rows=%d",
             len(X_tr), len(X_val), len(X_test))

    # Save processed splits
    X_tr.to_csv(cfg.processed_data_dir / "X_train.csv", index=False)
    X_val.to_csv(cfg.processed_data_dir / "X_val.csv", index=False)
    X_test.to_csv(cfg.processed_data_dir / "X_test.csv", index=False)
    y_tr.to_csv(cfg.processed_data_dir / "y_train.csv", index=False)
    y_val.to_csv(cfg.processed_data_dir / "y_val.csv", index=False)
    y_test.to_csv(cfg.processed_data_dir / "y_test.csv", index=False)

    # -------- 4. Transformer (PCA optional) --------
    log.info("[4/11] Data Transformation")
    transformer = DataTransformer(features=cfg.features,
                                  params_cfg=cfg.params)
    Xt_tr, trans_art = transformer.fit_transform(X_tr, cfg.preprocessing_dir)
    # Note: we keep PCA OFF by default (ternary-encoded data), so Xt_tr == X_tr

    # -------- 5. Feature Engineering --------
    log.info("[5/11] Feature Engineering")
    # Compute MI on raw features for interaction selection
    from sklearn.feature_selection import mutual_info_classif
    mi_raw = pd.Series(
        mutual_info_classif(X_tr, y_tr, random_state=42,
                             discrete_features=True),
        index=X_tr.columns).sort_values(ascending=False)
    log.info("Top-5 MI features: %s", list(mi_raw.head(5).items()))

    X_eng_tr, eng = fit_engineer(X_tr, cfg.features, mi_raw,
                                 cfg.params, cfg.features_dir)
    # Apply engineer to val/test as well
    X_eng_val = eng.transform(X_val)
    X_eng_test = eng.transform(X_test)
    X_eng_tr.to_csv(cfg.features_dir / "X_train_engineered.csv", index=False)
    X_eng_val.to_csv(cfg.features_dir / "X_val_engineered.csv", index=False)
    X_eng_test.to_csv(cfg.features_dir / "X_test_engineered.csv", index=False)

    # -------- 6. Feature Selection --------
    log.info("[6/11] Feature Selection")
    selector = FeatureSelector(
        original_features=cfg.features,
        top_k=int(cfg.params.feature_selection.top_k),
        keep_original=bool(cfg.params.feature_selection.keep_original_features),
        random_state=cfg.random_state,
    )
    X_sel_tr, sel_report = selector.select(
        X_eng_tr, y_tr, cfg.features_dir)
    X_sel_val = selector.transform(X_eng_val)
    X_sel_test = selector.transform(X_eng_test)
    log.info("Selected features (%d): %s",
             len(selector.selected_features_),
             selector.selected_features_[:10])

    # -------- 7. Clustering (optional / interpretive) --------
    log.info("[7/11] Unsupervised Clustering (optional/interpretive)")
    if bool(cfg.params.clustering.enabled):
        try:
            clusterer = ClusterTrainer(params_cfg=cfg.params)
            cluster_report, _labels_per_algo = clusterer.train_all(
                X_sel_tr, cfg.clusters_dir)
            cluster_names = name_clusters(cluster_report, X_sel_tr,
                                         selector.selected_features_)
            write_json(cfg.clusters_dir / "cluster_names.json", cluster_names)
        except Exception as exc:
            log.warning("Clustering skipped due to error: %s", exc)

    # -------- 8. Supervised Classification --------
    log.info("[8/11] Supervised Classification Training")
    trainer = ClassificationTrainer(params_cfg=cfg.params,
                                    model_cfg=cfg.model)
    results = trainer.train_all(X_sel_tr, y_tr, cfg.trained_models_dir)
    log.info("Trained %d supervised models", len(results))

    # AutoGluon (optional)
    log.info("[8b/11] AutoGluon TabularPredictor")
    ag_predictor = None
    try:
        if bool(cfg.params.classification.autogluon.enabled):
            ag_predictor = trainer.train_autogluon(
                X_sel_tr, y_tr, cfg.autogluon_dir)
    except Exception as exc:
        log.warning("AutoGluon skipped: %s", exc)

    # -------- 9. Evaluation --------
    log.info("[9/11] Evaluation")
    eval_results, cmp_df = evaluate_all(
        results, X_sel_test, y_test, cfg.evaluations_dir,
        autogluon_predictor=ag_predictor)
    cmp_df.to_csv(cfg.evaluations_dir / "model_comparison.csv", index=False)
    log.info("\n%s", cmp_df.to_string(index=False))

    best = select_best(eval_results)
    if best is None:
        log.error("No models trained successfully.")
        return

    log.info("Best model: %s  recall_phishing=%.4f roc_auc=%.4f",
             best.name, best.recall_phishing, best.roc_auc)

    # -------- 10. Save Best Model --------
    log.info("[10/11] Saving best model & metadata")
    is_xgb_like = best.name in {"XGBoost", "LightGBM", "CatBoost"}
    best_model_obj = results[best.name].model if best.name in results else None
    if best_model_obj is None and ag_predictor is not None and \
            best.name == "AutoGluon":
        best_model_obj = ag_predictor
    if best_model_obj is None:
        # Fallback: VotingEnsemble
        best_model_obj = results.get("VotingEnsemble", None)
    if best_model_obj is None:
        log.error("Could not locate best model object; aborting save.")
        return

    save_joblib(best_model_obj,
                cfg.trained_models_dir / "best_model.joblib", compress=3)
    write_json(cfg.trained_models_dir / "best_model_metadata.json", {
        "name": best.name,
        "is_xgb_like": is_xgb_like,
        "metrics": {
            "accuracy": best.accuracy, "precision": best.precision,
            "recall": best.recall, "f1": best.f1, "roc_auc": best.roc_auc,
            "pr_auc": best.pr_auc, "mcc": best.mcc,
            "log_loss": best.log_loss,
            "recall_phishing": best.recall_phishing,
            "confusion_matrix": best.confusion_matrix,
        },
        "selected_features": selector.selected_features_,
        "fit_time_seconds": best.fit_time_seconds,
    })

    # -------- 11. SHAP explainability --------
    log.info("[11/11] SHAP explainability")
    try:
        # For ensembles (Voting/Stacking) SHAP's TreeExplainer doesn't work.
        # Fall back to a tree-based model from the trained set.
        shap_model = best_model_obj
        shap_model_name = best.name
        tree_like_names = {"RandomForest", "ExtraTrees", "XGBoost", "LightGBM",
                           "CatBoost", "HistGradientBoosting",
                           "GradientBoosting"}
        if best.name not in tree_like_names:
            # Prefer XGBoost -> LightGBM -> CatBoost -> RF -> ExtraTrees
            for preferred in ["XGBoost", "LightGBM", "CatBoost",
                              "RandomForest", "ExtraTrees"]:
                if preferred in results:
                    shap_model = results[preferred].model
                    shap_model_name = preferred
                    log.info("Best model %s is not tree-compatible; using %s "
                             "for SHAP explainability", best.name, preferred)
                    break
        explainer = GlobalExplainer(
            model=shap_model,
            background=X_sel_tr,
            feature_names=selector.selected_features_,
        )
        importance = explainer.feature_importance(sample_size=100)
        write_json(cfg.shap_dir / "global_feature_importance.json",
                   importance.to_dict())
        write_json(cfg.shap_dir / "shap_metadata.json", {
            "model_used": shap_model_name,
            "best_model": best.name,
            "n_background_samples": 100,
        })
        log.info("Top-5 SHAP features (model=%s): %s",
                 shap_model_name, list(importance.head(5).items()))
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)

    # -------- Save prediction pipeline wrapper (not the full objects) --------
    # The Streamlit app loads FE/Selector/Model separately and constructs
    # a PredictionPipeline itself.

    # -------- Final summary --------
    elapsed = time.time() - t_start
    log.info("=" * 72)
    log.info("Training complete in %.1fs", elapsed)
    log.info("Best model: %s (recall_phishing=%.4f, roc_auc=%.4f)",
             best.name, best.recall_phishing, best.roc_auc)
    log.info("Artifacts saved under: %s", to_abs("artifacts"))
    log.info("=" * 72)


if __name__ == "__main__":
    main()
