"""
tests/test_pipeline.py
----------------------
Pytest suite for the Phishing Detection project.

Run with:
    pytest tests/test_pipeline.py -v

Each test exercises a single pipeline stage and verifies the artifact
contract (correct shape, correct schema, no missing values, etc.).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    get_config, setup_logging, save_joblib, load_joblib,
    map_numeric_to_label, map_label_to_numeric,
)
from src.data import DataIngestion, DataValidation
from src.features import FeatureEngineer, FeatureSelector, DataTransformer
from src.prediction import PredictionPipeline

setup_logging()
CFG = get_config()


# -------------------- Fixtures -------------------- #

@pytest.fixture(scope="session")
def ingestion_artifact():
    ing = DataIngestion(CFG.raw_data_dir,
                       train_file=CFG.config.files.train_data,
                       test_file=CFG.config.files.test_data)
    return ing.ingest()


@pytest.fixture(scope="session")
def loaded_fe():
    return FeatureEngineer.load(CFG.features_dir / "feature_engineer.joblib")


@pytest.fixture(scope="session")
def loaded_fs():
    return load_joblib(CFG.features_dir / "feature_selector.joblib")


@pytest.fixture(scope="session")
def loaded_best_model():
    return load_joblib(CFG.trained_models_dir / "best_model.joblib")


# -------------------- Tests -------------------- #

class TestConfig:
    def test_configs_load(self):
        assert CFG.config is not None
        assert CFG.params is not None
        assert CFG.schema is not None
        assert CFG.prediction_schema is not None
        assert CFG.model is not None
        assert CFG.logging is not None

    def test_features_count(self):
        assert len(CFG.features) == 30

    def test_target_name(self):
        assert CFG.target_name == "Result"

    def test_random_state(self):
        assert CFG.random_state == 42

    def test_paths_resolved(self):
        assert CFG.raw_data_dir.is_absolute()
        assert CFG.trained_models_dir.is_absolute()


class TestIngestion:
    def test_train_shape(self, ingestion_artifact):
        assert ingestion_artifact.n_rows_train == 11055
        assert ingestion_artifact.n_features == 30

    def test_test_shape(self, ingestion_artifact):
        assert ingestion_artifact.n_rows_test == 11055

    def test_train_has_target(self, ingestion_artifact):
        assert "Result" in ingestion_artifact.train_df.columns

    def test_test_no_target(self, ingestion_artifact):
        assert "Result" not in ingestion_artifact.test_df.columns


class TestValidation:
    def test_validation_passes(self, ingestion_artifact):
        validator = DataValidation(
            schema_cfg=CFG.schema,
            features=CFG.features,
            target_name=CFG.target_name,
        )
        report = validator.validate(
            ingestion_artifact.train_df, ingestion_artifact.test_df,
            CFG.validated_data_dir,
        )
        assert report.is_valid is True

    def test_feature_domain(self, ingestion_artifact):
        for feat in CFG.features:
            unique = set(pd.unique(ingestion_artifact.train_df[feat]))
            assert unique.issubset({-1, 0, 1}), \
                f"Feature {feat} has values {unique - {-1, 0, 1}}"

    def test_target_domain(self, ingestion_artifact):
        unique = set(pd.unique(ingestion_artifact.train_df["Result"]))
        assert unique.issubset({-1, 1})

    def test_no_missing_values(self, ingestion_artifact):
        assert ingestion_artifact.train_df.isna().sum().sum() == 0
        assert ingestion_artifact.test_df.isna().sum().sum() == 0


class TestFeatureEngineer:
    def test_engineer_loaded(self, loaded_fe):
        assert loaded_fe is not None
        assert hasattr(loaded_fe, "transform")

    def test_engineer_expands_features(self, loaded_fe, ingestion_artifact):
        X = ingestion_artifact.train_df[CFG.features].head(100)
        X_eng = loaded_fe.transform(X)
        assert X_eng.shape[0] == 100
        assert X_eng.shape[1] > 30
        # All original features retained
        for f in CFG.features:
            assert f in X_eng.columns
        # Engineered features present
        for eng_feat in ["risk_score_sum", "n_suspicious", "phishing_ratio"]:
            assert eng_feat in X_eng.columns


class TestFeatureSelector:
    def test_selector_loaded(self, loaded_fs):
        assert loaded_fs is not None
        assert hasattr(loaded_fs, "selected_features_")
        assert len(loaded_fs.selected_features_) > 0

    def test_selector_keeps_originals(self, loaded_fs):
        for f in CFG.features:
            assert f in loaded_fs.selected_features_, \
                f"Original feature {f} not in selected set"

    def test_selector_transform_shape(self, loaded_fs, loaded_fe,
                                      ingestion_artifact):
        X = ingestion_artifact.train_df[CFG.features].head(100)
        X_eng = loaded_fe.transform(X)
        X_sel = loaded_fs.transform(X_eng)
        assert X_sel.shape[1] == len(loaded_fs.selected_features_)
        assert X_sel.shape[0] == 100


class TestLabelMapping:
    def test_numeric_to_string(self):
        assert map_numeric_to_label(-1) == "phising"
        assert map_numeric_to_label(1) == "safe"

    def test_string_to_numeric(self):
        assert map_label_to_numeric("phising") == -1
        assert map_label_to_numeric("safe") == 1
        # Common misspelling should also work
        assert map_label_to_numeric("phishing") == -1
        assert map_label_to_numeric("legitimate") == 1


class TestPredictionPipeline:
    def test_pipeline_constructs(self, loaded_fe, loaded_fs,
                                 loaded_best_model):
        pipe = PredictionPipeline(
            feature_engineer=loaded_fe,
            feature_selector=loaded_fs,
            model=loaded_best_model,
            is_xgb_like=False,
        )
        assert pipe is not None

    def test_single_prediction_returns_label(self, loaded_fe, loaded_fs,
                                             loaded_best_model,
                                             ingestion_artifact):
        pipe = PredictionPipeline(
            feature_engineer=loaded_fe,
            feature_selector=loaded_fs,
            model=loaded_best_model,
        )
        # Use the first training row as a test record
        row = ingestion_artifact.train_df[CFG.features].iloc[0].to_dict()
        out = pipe.predict_single(row)
        assert out.label in {"phising", "safe"}
        assert 0.0 <= out.confidence <= 1.0
        assert 0.0 <= out.p_phishing <= 1.0
        assert 0.0 <= out.p_safe <= 1.0
        assert out.raw_prediction in {-1, 1}
        # Confidence matches the predicted class
        if out.label == "phising":
            assert out.confidence == pytest.approx(out.p_phishing,
                                                    rel=1e-3)
        else:
            assert out.confidence == pytest.approx(out.p_safe,
                                                    rel=1e-3)

    def test_batch_prediction_schema(self, loaded_fe, loaded_fs,
                                     loaded_best_model,
                                     ingestion_artifact):
        pipe = PredictionPipeline(
            feature_engineer=loaded_fe,
            feature_selector=loaded_fs,
            model=loaded_best_model,
        )
        X_batch = ingestion_artifact.test_df[CFG.features].head(50)
        out = pipe.predict_batch(X_batch)
        # Output schema: 30 features + Result + confidence + p_phishing
        assert "Result" in out.columns
        assert "confidence" in out.columns
        assert "probability_phishing" in out.columns
        assert len(out) == 50
        # Result values are strings
        assert set(out["Result"].unique()).issubset({"phising", "safe"})
        # Confidence in [0,1]
        assert (out["confidence"] >= 0).all()
        assert (out["confidence"] <= 1).all()


class TestArtifacts:
    """Verify all expected artifacts exist after training.py ran."""

    REQUIRED_FILES = [
        "artifacts/data/raw/phising_08012020_120000.csv",
        "artifacts/data/raw/phisingtest.csv",
        "artifacts/data/raw/predicted_file.csv",
        "artifacts/data/validated/validation_report.json",
        "artifacts/features/feature_engineer.joblib",
        "artifacts/features/feature_selector.joblib",
        "artifacts/clusters/cluster_report.json",
        "artifacts/models/trained_models/best_model.joblib",
        "artifacts/models/trained_models/best_model_metadata.json",
        "artifacts/evaluations/eval_results.json",
        "artifacts/evaluations/model_comparison.csv",
        "artifacts/shap/global_feature_importance.json",
    ]

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_artifact_exists(self, rel_path):
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Missing artifact: {rel_path}"

    def test_eval_results_have_all_models(self):
        path = CFG.evaluations_dir / "eval_results.json"
        with path.open() as fh:
            data = json.load(fh)
        names = {r["name"] for r in data["results"]}
        expected = {"RandomForest", "ExtraTrees", "XGBoost", "LightGBM",
                    "CatBoost", "HistGradientBoosting", "GradientBoosting",
                    "AdaBoost", "Bagging", "VotingEnsemble",
                    "StackingEnsemble"}
        # AutoGluon is optional
        missing = expected - names
        assert not missing, f"Missing model evaluations: {missing}"

    def test_cluster_report_has_results(self):
        path = CFG.clusters_dir / "cluster_report.json"
        with path.open() as fh:
            data = json.load(fh)
        assert "results" in data
        assert len(data["results"]) >= 5  # At least 5 clustering algos ran
        assert "best_algorithm" in data


class TestSecurity:
    """Verify the per-feature security tips are well-formed."""

    def test_all_features_have_tips(self):
        tips = CFG.prediction_schema.feature_security_tips
        for feat in CFG.features:
            assert feat in tips, f"Missing security tips for {feat}"
            for value in ["-1", "0", "1"]:
                assert value in tips[feat], \
                    f"Missing tip for {feat}={value}"


if __name__ == "__main__":
    # Allow running without pytest for quick smoke tests
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
