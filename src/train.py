"""Train the phishing URL classifier.

Trains multiple models on the uploaded Dataset.csv, picks the best one by
phishing-recall + accuracy + F1, and saves the full inference pipeline plus
evaluation metrics to artifacts/.

Usage
-----
    python /home/z/my-project/phishing_app/src/train.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Local import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import Preprocessor

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Dataset.csv"
# Fall back to original upload path if local copy is missing
if not DATA_PATH.exists():
    DATA_PATH = Path("/home/z/my-project/upload/Dataset.csv")
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "models").mkdir(exist_ok=True)
(ARTIFACTS_DIR / "metrics").mkdir(exist_ok=True)
(ARTIFACTS_DIR / "shap").mkdir(exist_ok=True)

# Class labels (training dataset convention)
LABEL_LEGIT = 0
LABEL_PHISH = 1


def load_data() -> pd.DataFrame:
    """Load and clean the raw dataset."""
    print(f"[1/7] Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Initial shape: {df.shape}")

    # Drop the 14 rows with missing values.
    df = df.dropna().reset_index(drop=True)
    print(f"      After dropping NaNs: {df.shape}")

    # Sanity-check label values.
    assert set(df["label"].unique()).issubset({0, 1}), "Unexpected label values"
    print(f"      Class distribution: {df['label'].value_counts().to_dict()}")
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split."""
    print("[2/7] Splitting data (stratified 80/20) ...")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].copy()
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_model_zoo() -> dict:
    """Return the dictionary of candidate classifiers.

    Each entry is (display_name, callable).  The callable takes no args
    and returns a fresh, un-fitted estimator configured for class
    imbalance.
    """
    # scale_pos_weight for the boosting models = neg / pos
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=100000 / 16600,  # ~6.0
            eval_metric="logloss", n_jobs=-1, random_state=42,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.08,
            auto_class_weights="Balanced",
            verbose=False, random_state=42,
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            class_weight="balanced", random_state=42,
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42,
        ),
        "Bagging": lambda: BaggingClassifier(
            n_estimators=30, n_jobs=-1, random_state=42,
        ),
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
        ]),
    }


def evaluate_model(name: str, estimator, X_test, y_test) -> dict:
    """Compute the full evaluation metric suite."""
    y_pred = estimator.predict(X_test)
    try:
        y_proba = estimator.predict_proba(X_test)[:, 1]
    except Exception:
        # Some pipelines may not have predict_proba; fall back to decision_function
        y_proba = estimator.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_phish": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_phish": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_phish": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1])),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train_all(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame, Preprocessor]:
    """Train every candidate, evaluate, and return (pipelines, metrics_df, preprocessor)."""
    print("[3/7] Fitting preprocessor ...")
    preprocessor = Preprocessor(top_n=50)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"      Transformed train shape: {X_train_t.shape}")

    print("[4/7] Training candidate models ...")
    zoo = build_model_zoo()
    fitted_pipelines: dict = {}
    rows: list[dict] = []

    for name, factory in zoo.items():
        t0 = time.time()
        print(f"      - {name} ...", end=" ", flush=True)
        estimator = factory()
        try:
            estimator.fit(X_train_t, y_train)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        elapsed = time.time() - t0
        metrics = evaluate_model(name, estimator, X_test_t, y_test)
        metrics["train_seconds"] = round(elapsed, 2)
        rows.append(metrics)
        fitted_pipelines[name] = estimator
        print(
            f"acc={metrics['accuracy']:.4f}  "
            f"recall_phish={metrics['recall_phish']:.4f}  "
            f"f1={metrics['f1_phish']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["recall_phish", "f1_phish", "accuracy"], ascending=False
    ).reset_index(drop=True)
    return fitted_pipelines, metrics_df, preprocessor


def pick_best(metrics_df: pd.DataFrame) -> str:
    """Pick the best model name.

    We prioritize phishing recall (missing a phishing site is the worst
    failure), then F1, then accuracy.
    """
    return str(metrics_df.iloc[0]["model"])


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    fitted, metrics_df, preprocessor = train_all(X_train, X_test, y_train, y_test)

    best_name = pick_best(metrics_df)
    best_model = fitted[best_name]
    print(f"[5/7] Best model: {best_name}")

    # Persist artifacts.
    print("[6/7] Saving artifacts ...")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "models" / "preprocessor.joblib")
    joblib.dump(best_model, ARTIFACTS_DIR / "models" / "best_model.joblib")
    joblib.dump(fitted, ARTIFACTS_DIR / "models" / "all_models.joblib")

    metrics_df.to_csv(ARTIFACTS_DIR / "metrics" / "model_comparison.csv", index=False)
    metrics_df.to_json(ARTIFACTS_DIR / "metrics" / "model_comparison.json", orient="records", indent=2)

    # Save per-model ROC + confusion for the dashboard.
    print("[7/7] Saving per-model ROC + confusion artifacts ...")
    X_test_t = preprocessor.transform(X_test)
    roc_data: dict = {}
    cm_data: dict = {}
    for name, est in fitted.items():
        try:
            proba = est.predict_proba(X_test_t)[:, 1]
        except Exception:
            proba = est.decision_function(X_test_t)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                          "auc": float(roc_auc_score(y_test, proba))}
        y_pred = est.predict(X_test_t)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_data[name] = cm.tolist()

    with open(ARTIFACTS_DIR / "metrics" / "roc_curves.json", "w") as f:
        json.dump(roc_data, f)
    with open(ARTIFACTS_DIR / "metrics" / "confusion_matrices.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    # Save best-model metadata
    best_metrics = metrics_df[metrics_df["model"] == best_name].iloc[0].to_dict()
    best_meta = {
        "best_model": best_name,
        "metrics": best_metrics,
        "feature_columns": preprocessor.feature_names(),
        "n_features": len(preprocessor.feature_names()),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with open(ARTIFACTS_DIR / "metrics" / "best_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)

    print("\n=== Training summary ===")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")

    # ----- SHAP explainer for the best model -----
    try:
        print("\n[shap] Computing SHAP TreeExplainer for the best model ...")
        import shap
        # Sample for speed
        sample = X_test_t.sample(min(2000, len(X_test_t)), random_state=42)
        if best_name in {"RandomForest", "ExtraTrees", "XGBoost", "LightGBM",
                         "CatBoost", "GradientBoosting", "HistGradientBoosting",
                         "AdaBoost", "Bagging"}:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_test_t, 100))
        shap_values = explainer.shap_values(sample)
        # For binary classification with shape (n_samples, n_features, 2),
        # keep only the phishing class contribution.
        if isinstance(shap_values, list):
            sv_phish = shap_values[1]
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            sv_phish = shap_values[:, :, 1]
        else:
            sv_phish = shap_values

        np.save(ARTIFACTS_DIR / "shap" / "shap_values.npy", sv_phish)
        sample.to_csv(ARTIFACTS_DIR / "shap" / "shap_sample.csv", index=False)
        joblib.dump(explainer, ARTIFACTS_DIR / "shap" / "explainer.joblib")
        print(f"[shap] Saved shap_values shape={sv_phish.shape}")
    except Exception as exc:
        print(f"[shap] SKIPPED: {exc}")


if __name__ == "__main__":
    main()
